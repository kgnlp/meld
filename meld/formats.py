"""Utilities for reading and writing the unified MELD parquet format."""

import itertools
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import IO, Self

import polars as pl
import pyarrow
from polars import LazyFrame
from pyarrow import Field, NativeFile, RecordBatch, Schema
from pyarrow.parquet import ParquetFile, ParquetWriter
from pydantic import TypeAdapter
from tokenizers import Tokenizer

from meld.document import SEQUENCE_TYPES, Annotation, BIOField, NERDocument
from meld.manifest import Metadata

METADATA_FILENAME = "meld_metadata.json"
PROCESSED_DIRECTORY = "meld"

DEFAULT_TAGSET = "ner"


def default_tagset[T](tags: list[T]) -> dict[str, list[T]]:
    """
    Wrap tags in a default tagset dictionary.

    Args:
        tags: List of tags.

    Returns:
        Dictionary with the default tagset name mapping to the tags.
    """

    return {DEFAULT_TAGSET: tags}


def tagset_field(field_name: str) -> Field:
    """
    Create a PyArrow field definition for a tagset column.

    Args:
        field_name: The name of the field to create.

    Returns:
        A PyArrow Field representing the tagset structure.
    """

    return pyarrow.field(
        field_name,
        pyarrow.list_(
            pyarrow.struct(
                [
                    pyarrow.field("label", pyarrow.string(), nullable=False),
                    pyarrow.field(
                        "spans",
                        pyarrow.list_(
                            pyarrow.struct(
                                [
                                    pyarrow.field("start", pyarrow.uint32(), nullable=False),
                                    pyarrow.field("stop", pyarrow.uint32(), nullable=False),
                                ]
                            )
                        ),
                        nullable=False,
                    ),
                ]
            ),
        ),
    )


def iob_field(tagset_name: str) -> Field:
    """
    Create a PyArrow field definition for an IOB tagset column.

    Args:
        tagset_name: The name of the tagset to create the IOB field for.

    Returns:
        A PyArrow Field representing the IOB tagset structure.
    """

    return pyarrow.field(f"{tagset_name}_iob", pyarrow.list_(pyarrow.string()), nullable=False)


TEXT_ONLY_SCHEMA = pyarrow.schema(
    [
        pyarrow.field("sequence_id", pyarrow.binary(), nullable=False),
        pyarrow.field("document_index", pyarrow.uint64(), nullable=False),
        pyarrow.field("document_position", pyarrow.uint64(), nullable=False),
        pyarrow.field("sequence_type", pyarrow.uint8(), nullable=False),
        pyarrow.field("text", pyarrow.string(), nullable=False),
        pyarrow.field("space_after", pyarrow.string(), nullable=False),
    ],
    metadata={"sequence_types": json.dumps(["sentence", "passage"])},
)


FULL_SCHEMA = TEXT_ONLY_SCHEMA.append(pyarrow.field("tokens", pyarrow.list_(pyarrow.string()), nullable=False))


# Random UUID v4 namespace for generated sequence_ids
_UUID_NAMESPACE = uuid.UUID("c9d3a311-7c27-4a86-b6fb-a52aa9f53b63")


class NERParquetWriter:
    """
    Writer for NER data in MELD parquet format.

    Args:
        dataset: Name of the dataset.
        subset: Name of the subset.
        split: Name of the split (e.g., train, test).
        writer: The PyArrow ParquetWriter instance to write to.
        schema: The arrow schema to use for writing.
    """

    _BUFFER_SIZE = 64

    def __init__(self, dataset: str, subset: str, split: str, writer: ParquetWriter, schema: Schema) -> None:
        self._sequence_id_prefix = f"{dataset} {subset} {split}".encode()
        self._writer = writer
        self._annotation_serializer = TypeAdapter(list[Annotation])
        self._tokens_serializer = TypeAdapter(list[BIOField])
        self._schema = schema
        # Parse tagsets from schema metadata, assuming tagset columns always appear last and in the given order
        self._tagsets = json.loads(schema.metadata[b"tagsets"])
        # Automatically detect whether the schema contains tokens and iob labels
        self._is_tokenized = schema.get_field_index("tokens") != -1

    def _initialize_buffer(self) -> None:
        """Initialize internal column buffers for batched writing."""
        self._sentence_buffer = arrays = [[] for _ in range(len(self._schema))]
        self._record_buffer = {
            "sequence_id": arrays[0],
            "document_index": arrays[1],
            "document_position": arrays[2],
            "sequence_type": arrays[3],
            "text": arrays[4],
            "space_after": arrays[5],
        }
        tagsets_start = 6

        if self._is_tokenized:
            self._record_buffer["tokens"] = arrays[tagsets_start]
            tagsets_start += 1

        for i, tagset in enumerate(self._tagsets, tagsets_start):
            self._record_buffer[tagset] = arrays[i]

        if self._is_tokenized:
            for i, tagset in enumerate(self._tagsets, tagsets_start + len(self._tagsets)):
                self._record_buffer[f"{tagset}_iob"] = arrays[i]

    def __enter__(self) -> Self:
        self._initialize_buffer()
        self._document_index = 0
        return self

    def write_document(self, document: NERDocument) -> None:
        """
        Write a document to the underlying parquet file.

        Args:
            document: The NER document to write.
        """

        tokenized = document.bio if document.bio is not None else itertools.repeat(None)
        arrays = self._sentence_buffer

        for document_position, (labeled_text, labeled_tokens) in enumerate(zip(document.spans, tokenized)):
            # Generate reproducible UUID v5 IDs for each sentence or passage in the data
            # NOTE: Assumes documents are written in a deterministic order!
            sequence_id = uuid.uuid5(
                _UUID_NAMESPACE,
                self._sequence_id_prefix
                + self._document_index.to_bytes(64, "little")
                + document_position.to_bytes(64, "little"),
            ).bytes

            arrays[0].append(sequence_id)
            arrays[1].append(self._document_index)
            arrays[2].append(document_position)
            arrays[3].append(SEQUENCE_TYPES[labeled_text.sequence_type])
            arrays[4].append(labeled_text.text)
            arrays[5].append(labeled_text.space_after)
            tagsets_start = 6

            if labeled_tokens is not None:
                arrays[tagsets_start].append(labeled_tokens.tokens)
                tagsets_start += 1

            for i, tagset in enumerate(self._tagsets, tagsets_start):
                arrays[i].append(self._annotation_serializer.dump_python(labeled_text.labels[tagset]))

            tagsets_start += len(self._tagsets)

            if labeled_tokens is not None:
                for i, tagset in enumerate(self._tagsets, tagsets_start):
                    arrays[i].append(self._tokens_serializer.dump_python(labeled_tokens.labels[tagset]))

        # Batches can be different lengths but must contain at least 64 sentences if possible
        if len(self._sentence_buffer[0]) > self._BUFFER_SIZE:
            self._writer.write_batch(RecordBatch.from_pydict(self._record_buffer, self._schema))
            self._initialize_buffer()

        self._document_index += 1

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        # Write final batch
        if self._sentence_buffer[0]:
            self._writer.write_batch(RecordBatch.from_pydict(self._record_buffer, self._schema))

    @classmethod
    @contextmanager
    def open(
        cls,
        dataset: str,
        subset: str,
        split: str,
        path: str | Path | IO | NativeFile,
        with_tokens: bool,
        tagsets: list[str],
    ) -> Iterator[Self]:
        """
        Open a NERParquetWriter as a context manager.

        Args:
            dataset: Name of the dataset.
            subset: Name of the subset.
            split: Name of the split.
            path: Path to to which the parquet file will be written.
            with_tokens: Whether to include token columns in the output.
            tagsets: List of tagsets to include.

        Yields:
            NERParquetWriter instance.
        """

        schema = FULL_SCHEMA if with_tokens else TEXT_ONLY_SCHEMA
        schema = FULL_SCHEMA if with_tokens else TEXT_ONLY_SCHEMA
        for tagset in tagsets:
            schema = schema.append(tagset_field(tagset))

        schema = schema.with_metadata({"tagsets": json.dumps(tagsets), **schema.metadata})

        if with_tokens:
            for tagset in tagsets:
                schema = schema.append(iob_field(tagset))

        with (
            ParquetWriter(path, schema, compression="zstd") as parquet_writer,
            cls(dataset, subset, split, parquet_writer, schema) as writer,
        ):
            yield writer


@dataclass(slots=True)
class SplitMetadata:
    """
    Metadata for a split in a dataset.

    Attributes:
        path: Path to the split data.
        document_count: Number of documents in the split.
        sequence_count: Number of sequences in the split.
        creation_metadata: Metadata containing information on licensing,
            domains, sources, and the annotation process that apply to
            this split
    """

    path: Path
    document_count: int = 0
    sequence_count: int = 0
    creation_metadata: Metadata = field(default_factory=Metadata)


@dataclass(slots=True)
class Split:
    """
    Metadata for a specific data split within a dataset subset.

    Attributes:
        name: Name of this split (e.g. "train", "validation", or "test")
        dataset_name: Name of the dataset this split belongs to
        language: Language of the data in this split
        tagsets: List of tagset names available in this split
        metadata: Metadata about the split including size and creation information
        labels: Dictionary mapping tagset names to their respective label lists
        bio_labels: Optional dictionary mapping tagset names to BIO-formatted label lists
    """

    name: str
    dataset_name: str
    language: str
    tagsets: list[str]
    metadata: SplitMetadata
    labels: dict[str, list[str]]
    bio_labels: dict[str, list[str]] | None = None


@dataclass(slots=True)
class SubsetMetadata:
    """
    Metadata for a subset of a dataset.

    Attributes:
        language: Language of the subset.
        pre_tokenized: Whether the subset is pre-tokenized.
        tagsets: Tag sets available in the subset.
        labels: List of entity labels in the dataset.
        splits: Metadata of the splits in the subset by split name.
        bio_labels: List of BIO labels associated with the dataset if available.
    """

    language: str
    pre_tokenized: bool
    tagsets: list[str]
    labels: dict[str, list[str]]
    splits: dict[str, SplitMetadata] = field(default_factory=dict)
    bio_labels: dict[str, list[str]] | None = None

    @classmethod
    def _default_with_language(cls, language: str) -> Self:
        """
        Creates a default SubsetMetadata instance with the given language, no labels, and sets pre_tokenized to False.

        Args:
            language: Language of the subset.

        Returns:
            Default SubsetMetadata instance.
        """

        return cls(language, False, [DEFAULT_TAGSET], {})


type SubsetHierarchy = dict[str, SubsetMetadata | SubsetHierarchy]


@dataclass(slots=True)
class _SubsetWithMetadata:
    """
    Represents a subset containing its metadata and path in the subset hierarchy.

    Attributes:
        metadata: Metadata containing information about the subset.
        hierarchy: A flat representation of the hierarchy, including the
            subset name and names of all its parent subsets
    """

    metadata: SubsetMetadata
    hierarchy: list[str]

    @property
    def splits(self) -> dict[str, SplitMetadata]:
        """
        Returns the splits metadata.

        Returns:
            A dictionary of split names and their corresponding
            metadata.
        """

        return self.metadata.splits

    def get_split_metadata(self, name: str) -> SplitMetadata | None:
        """
        Retrieves metadata for a specific split.

        Args:
            name: The name of the split to retrieve metadata for.

        Returns:
            The metadata for the specified split, or None if it does not
            exist.
        """

        return self.metadata.splits.get(name)

    def __contains__(self, split_name: str) -> bool:
        """
        Checks if a specific split exists in the subset.

        Args:
            split_name: The name of the split to check for.

        Returns:
            True if a split with the given name exists.
        """

        return split_name in self.metadata.splits


@dataclass(slots=True)
class DatasetMetadata:
    """
    Metadata of a dataset.

    Attributes:
        name: Name of the dataset.
        subsets: Dictionary mapping subset names to SubsetMetadata
            instances or further SubsetHierarchy levels.
        languages: Set of languages included in the dataset.
        main_splits: Main SplitMetadata instance if available.
    """

    name: str
    subsets: SubsetHierarchy
    languages: set[str]
    main_splits: SubsetMetadata | None = None

    def _iter_subsets(self) -> Iterator[_SubsetWithMetadata]:
        """
        Iterates over all subsets in the hierarchy recursively.

        Yields:
            SubsetInformation for each subset.
        """

        if self.main_splits is not None:
            yield _SubsetWithMetadata(self.main_splits, [])

        # BFS through the subset hierarchy
        subset_queue = [([name], hierarchy) for name, hierarchy in self.subsets.items()]
        while subset_queue:
            path, current = subset_queue.pop()
            if isinstance(current, SubsetMetadata):
                yield _SubsetWithMetadata(current, path)
                continue

            for name, hierarchy in current.items():
                subset_queue.append((path + [name], hierarchy))

    @classmethod
    def from_json(cls, manifest_path: Path) -> Self:
        """
        Creates an instance of `DatasetMetadata` from a JSON file.

        Args:
            manifest_path: Path to the JSON metadata file.

        Returns:
            An instance of `DatasetMetadata`.
        """

        metadata_parser = TypeAdapter(cls)
        with manifest_path.open("rb") as file:
            return metadata_parser.validate_json(file.read())

    def dump(self, directory: Path) -> None:
        """
        Write the dataset metadata to a JSON file in the specified directory.

        Args:
            directory: Directory into which the metadata should be
                written.
        """

        # Write metadata for the fully processed dataset
        metadata_serializer = TypeAdapter(self.__class__)
        with (directory / METADATA_FILENAME).open("wb") as file:
            file.write(metadata_serializer.dump_json(self))


type AnyPath = PathLike | str


@dataclass(slots=True)
class Subset(_SubsetWithMetadata):
    """
    Represents a subset of a dataset.

    Attributes:
        metadata: Metadata containing information about the subset.
        hierarchy: A flat representation of the hierarchy, including the
            subset name and names of all its parent subsets
        dataset_path: Path to the main dataset directory.
    """

    dataset_path: Path

    def split_path(self, name: str) -> Path:
        """
        Construct the path to a specific split of the dataset.

        Args:
            name: Name of the split.

        Returns:
            Path to the split.
        """

        return self.dataset_path / self.metadata.splits[name].path

    def open_split(self, name: str) -> ParquetFile:
        """
        Open a given split for reading as a `pyarrow.ParquetFile`.

        Args:
            name: Name of the split to open.

        Returns:
            A `pyarrow.ParquetFile` containing the split data.
        """

        return ParquetFile(self.split_path(name))

    def scan_split(self, name: str) -> LazyFrame:
        """
        Scan the split into a lazy Polars DataFrame.

        Args:
            name: Name of the split to scan.

        Returns:
            A Polars LazyFrame containing the split data.
        """

        return pl.scan_parquet(self.split_path(name))


@dataclass(slots=True)
class Dataset:
    """
    Represents a full dataset made up of one or more subsets.

    Attributes:
        path: Path to the main local dataset directory.
        metadata: Metadata for the dataset.
        subsets: List of subsets in the dataset.
    """

    path: Path
    metadata: DatasetMetadata
    subsets: dict[tuple[str, ...], Subset] = field(init=False)

    def __post_init__(self) -> None:
        """Initializes the subsets from the dataset metadata."""
        self.subsets = {
            tuple(subset_info.hierarchy): Subset(subset_info.metadata, subset_info.hierarchy, self.path)
            for subset_info in self.metadata._iter_subsets()
        }

    def __iter__(self) -> Iterator[Subset]:
        """
        Iterate over all available subsets.

        Yields:
            All subsets of the dataset.
        """

        yield from self.subsets.values()

    @classmethod
    def load(cls, benchmark_path: Path, dataset_name: str) -> Self:
        """
        Load a dataset from the given benchmark path and dataset name.

        Args:
            benchmark_path: The root directory of the locally processed
                benchmark.
            dataset_name: The name of the dataset to load.

        Returns:
            A `Dataset` instance representing the given dataset.
        """

        dataset_directory = _local_processed_directory(benchmark_path) / dataset_name
        return cls(dataset_directory, DatasetMetadata.from_json(dataset_directory / METADATA_FILENAME))


def _local_processed_directory(data_directory: AnyPath) -> Path:
    """
    Return the path to the local processed data directory.

    Args:
        data_directory: The base data directory.

    Returns:
        The path to the local processed data directory.
    """

    return Path(data_directory) / PROCESSED_DIRECTORY


def local_dataset_names(data_directory: AnyPath) -> Iterator[str]:
    """
    Yields names of datasets located in the specified directory.

    Args:
        data_directory: Path to the directory containing dataset
            folders.
    Yields:
        Name of each dataset.
    """

    for path in _local_processed_directory(data_directory).iterdir():
        if (path / METADATA_FILENAME).is_file():
            yield path.name


def _local_dataset_metadata(data_directory: AnyPath) -> Iterator[DatasetMetadata]:
    """
    Yields DatasetMetadata instances for all datasets located in the specified directory.

    Args:
        data_directory: Path to the directory containing dataset
            folders.
    Yields:
        DatasetMetadata instance for each dataset.
    """

    for path in _local_processed_directory(data_directory).iterdir():
        manifest_path = path / METADATA_FILENAME
        if not manifest_path.is_file():
            continue

        yield DatasetMetadata.from_json(manifest_path)


def local_datasets(data_directory: AnyPath) -> Iterator[Dataset]:
    """
    Yields all locally downloaded datasets within a given directory.

    Args:
        data_directory: Path to the directory containing dataset
            metadata files.
    Yields:
        An iterator over Dataset instances.
    """

    processed_directory = _local_processed_directory(data_directory)
    for metadata in _local_dataset_metadata(data_directory):
        yield Dataset(processed_directory / metadata.name, metadata)


def _distributed_limit(limit: int, splits: int) -> list[int]:
    """
    Calculates distributed limits for a given limit and number of splits.

    Args:
        limit: Total limit of samples.
        splits: Number of split.

    Returns:
        List of integers representing the limit for each split.
    """

    split_size, remainder = divmod(limit, splits)
    return list(itertools.chain((split_size for _ in range(splits - 1)), (split_size + remainder,)))


def _merge_documents(documents: LazyFrame, merge_by_segments: bool = False) -> LazyFrame:
    """
    Merge documents that are split into multiple segments.

    Args:
        documents: Input LazyFrame with document indexed data.
        merge_by_segments: Whether to merge into segments via segment
            IDs or only by document index.

    Returns:
        Merged LazyFrame with combined documents.
    """

    if merge_by_segments:
        merged = documents.group_by("document_index", "segment_id", maintain_order=True)
    else:
        merged = documents.cast({"document_index": pl.Int64}).group_by_dynamic("document_index", every="1i")

    ner_column = [DEFAULT_TAGSET] if DEFAULT_TAGSET in documents.collect_schema() else []

    merged = (
        merged.agg(
            pl.col("sequence_id").first(),
            (pl.col("text") + pl.col("space_after")).str.join(""),
            *ner_column,
            # Add 1 to the lengths for every added space
            sentence_offset=(pl.col("text").str.len_chars() + pl.col("space_after").str.len_chars()).cum_sum(),
        )
        .with_row_index("chunk_index")
        .cast({"chunk_index": pl.Int64})
        # Shift the length sums by one and replace the offset of the first sentence with 0
        .with_columns(sentence_offset=pl.col("sentence_offset").list.shift(1).list.eval(pl.element().fill_null(0)))
    )

    # No labels need to be re-aligned with the joined text
    if not ner_column:
        return merged.select("sequence_id", "chunk_index", "text")

    entity_offsets = (
        merged.explode("ner", "sentence_offset")
        .explode("ner")
        .with_columns(pl.col("ner").struct.unnest())
        .with_row_index("label_index")
        .cast({"label_index": pl.Int64})
        .explode("spans")
        .with_columns("label", spans=pl.col("spans") + pl.col("sentence_offset"))
        .group_by_dynamic("label_index", every="1i")
        .agg(pl.col("label", "chunk_index", "sequence_id").first(), "spans")
        .select(
            "chunk_index",
            ner=pl.struct(
                label=pl.col("label"),
                spans=pl.col("spans"),
            ),
        )
        .group_by_dynamic("chunk_index", every="1i")
        .agg("ner")
        .with_columns(
            ner=pl.col("ner")
            .list.eval(pl.when(pl.element().struct["label"].is_null()).then(None).otherwise(pl.element()))
            .list.drop_nulls()
        )
    )

    return pl.concat(
        [merged.select("sequence_id", "chunk_index", "text"), entity_offsets.select("ner")],
        how="horizontal",
    )


def read_monolingual_sample(
    data_directory: AnyPath,
    language: str,
    samples_per_datasets: int,
    split_name: str = "train",
    tagset_config: dict[str, str] | None = None,
    merge_documents: bool = False,
    keep_documents_without_entities: bool = True,
    add_dataset_column: bool = True,
    target_num_tokens: int | None = None,
    aggregation_tokenizer: str = "google/gemma-3-27b-it",
) -> LazyFrame:
    """
    Reads monolingual sample data from the specified dataset.

    Args:
        data_directory: Path to the directory containing dataset
            folders.
        language: Language of the subset to process.
        samples_per_datasets: Maximum number of samples per dataset.
        split_name: Name of the split.
        tagset_config: Indicates which tagset to use for datasets with
            multiple tag sets for each sample. E.g. `{"Few-NERD":
            "fine"}` selects fine-grained tags from the `Few-NERD`
            dataset. This parameter is required if a dataset with
            multiple tagsets is encountered during sampling with the
            given configuration.
        merge_documents: Whether documents should be merged based on
            document_index.
        keep_documents_without_entities: Whether to keep documents
            without entities.
        add_dataset_column: Whether to add a dataset column to the
            resulting dataframe.
        target_num_tokens: If `merge_documents` is true, attempts to
            merge sentences or passages into documents only if the given
            number of tokens is not exceeded.
        aggregation_tokenizer: Tokenizer used for counting tokens if
            `target_num_tokens` is set.

    Returns:
        Polars LazyFrame containing the processed samples.

    Raises:
        KeyError: When a dataset has multiple tagsets but no tagset is
            selected via `tagset_config`.
        ValueError: When no samples are found for the specified
            language.
    """

    if merge_documents and target_num_tokens is not None:
        tokenizer = Tokenizer.from_pretrained(aggregation_tokenizer)
    else:
        tokenizer = None

    data_directory = Path(data_directory)
    dataset_samples = []
    for dataset in local_datasets(data_directory):
        subset_data = []
        for subset in dataset:
            if subset.metadata.language == language and split_name in subset:
                selected_tagset = None
                if len(subset.metadata.tagsets) > 1 and (
                    tagset_config is None or (selected_tagset := tagset_config.get(dataset.metadata.name)) is None
                ):
                    raise KeyError(
                        f"No tagset specified for {dataset.metadata.name} which contains multiple tagsets ({subset.metadata.tagsets})"
                    )

                documents = subset.scan_split(split_name).select(
                    "document_index", "sequence_id", "text", "ner" if selected_tagset is None else selected_tagset
                )
                if selected_tagset is not None:
                    documents = documents.rename({selected_tagset: "ner"})

                if tokenizer is not None:
                    documents = (
                        documents.with_columns(
                            tokens=pl.col("text").map_elements(
                                lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
                                pl.Int64,
                                skip_nulls=False,
                            )
                        )
                        .cast({"document_index": pl.Int64})
                        .group_by_dynamic("document_index", every="1i")
                        .agg(
                            pl.exclude("document_index"),
                            segment_id=pl.col("tokens").cum_sum() // target_num_tokens,
                        )
                        .explode(pl.exclude("document_index"))  # pyright: ignore
                    )

                if merge_documents:
                    documents = _merge_documents(documents, target_num_tokens is not None)

                if not keep_documents_without_entities:
                    documents = documents.filter(pl.col("ner").list.len() > 0)

                subset_data.append(documents)

        if subset_data:
            # NOTE: Sampling an equal fraction of each subset might lead to a total size that is smaller than the maximum
            # if subset sizes are unbalanced even if the maximum could be reached with another mix
            documents = pl.concat(
                [
                    subset.limit(subset_samples)
                    for subset, subset_samples in zip(
                        subset_data, _distributed_limit(samples_per_datasets, len(subset_data))
                    )
                ]
            )
            if add_dataset_column:
                documents = documents.with_columns(dataset=pl.lit(dataset.metadata.name))

            dataset_samples.append(documents)

    if not dataset_samples:
        raise ValueError(f"No data found for ISO-6393 language code {language}")

    return pl.concat(dataset_samples)


def drop_discontinuous_spans(dataset: LazyFrame) -> LazyFrame:
    """
    Drops discontinuous spans from a given dataset.

    Args:
        dataset: Polars LazyFrame with 'ner' column containing
            potentially discontinuous spans.

    Returns:
        Polars LazyFrame with discontinuous spans removed.
    """

    return dataset.with_columns(
        ner=pl.col("ner")
        .list.eval(pl.element().filter(pl.element().struct["spans"].list.len() == 1))
        .list.eval(pl.struct(label=pl.element().struct["label"], spans=pl.element().struct["spans"].list[0]))
    )
