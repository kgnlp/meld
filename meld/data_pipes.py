"""Data pipe system for downloading, processing and normalizing NER datasets."""

import contextlib
import hashlib
import itertools
import json
import logging
import operator
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import (
    Literal,
    Protocol,
    Self,
    runtime_checkable,
)
from urllib.parse import urlparse

import gdown
from datasets import BuilderConfig, Dataset, DatasetDict, get_dataset_config_names, load_dataset
from datasets.inspect import dataset_module_factory
from datasets.packaged_modules.csv.csv import CsvConfig
from datasets.packaged_modules.parquet.parquet import ParquetConfig
from huggingface_hub import snapshot_download
from langcodes import Language
from langcodes.tag_parser import LanguageTagError
from polars import DataFrame
from pydantic import TypeAdapter
from tqdm import tqdm

from meld import download
from meld.document import Annotation, NERDocument
from meld.download import extract
from meld.formats import (
    DEFAULT_TAGSET,
    DatasetMetadata,
    NERParquetWriter,
    SplitMetadata,
    SubsetHierarchy,
    SubsetMetadata,
)
from meld.manifest import (
    ConvertStep,
    DataPipeStep,
    DatasetConvertStep,
    DownloadStep,
    ExtractStep,
    GenericLoader,
    GitLoader,
    GitLoaderArguments,
    GitStep,
    GoogleDocsStep,
    HuggingfaceArguments,
    HuggingfaceLoader,
    Loader,
    MELDDataset,
    Metadata,
    NestedDataPipeStep,
    ReadSplitStep,
    SplitFiles,
    SplitStep,
    SubMetadata,
    SubsetDataFiles,
)
from meld.readers import (
    DatasetReader,
    DatasetSource,
    DataSource,
    FileSource,
    NamedTagSet,
    SplitData,
    SplitPaths,
    SubsetName,
    reader_from_config,
)
from meld.registry import Registry
from meld.tokenization import SentenceSplitter

logger = logging.getLogger("meld")


def _checked_update[K, V](dictionary: dict[K, V], other: dict[K, V]) -> dict[K, V]:
    """
    Merges two dictionaries, raising an error if keys overlap.

    Note:
        This function updates the larger of the two input dictionaries in-place.

    Args:
        dictionary: The target dictionary to update.
        other: Dictionary with keys to add.

    Returns:
        The updated dictionary.

    Raises:
        ValueError: If a key exists in both dictionaries.
    """

    # Reduce necessary number of checks if the "other" dictionary is significantly larger
    if len(other) > len(dictionary):
        dictionary, other = other, dictionary

    for key, value in other.items():
        if key in dictionary:
            raise ValueError(f"Key {key!r} exists in both dictionaries")
        dictionary[key] = value

    return dictionary


@dataclass(slots=True)
class _ProcessedDatasetMetadata:
    """
    Metadata for a processed dataset during data pipe execution.

    Attributes:
        subsets: Dictionary mapping subset names to metadata or nested
            hierarchies.
        main_splits: Metadata for root level splits.
        languages: Set of languages in the dataset.
    """

    subsets: SubsetHierarchy = field(default_factory=dict)
    main_splits: SubsetMetadata | None = None
    languages: set[str] = field(default_factory=set)

    def merge(self, other: Self) -> Self:
        """
        Merges this metadata with another.

        Args:
            other: Another _ProcessedDatasetMetadata instance.

        Returns:
            This instance with merged metadata.
        """

        # NOTE: Only merges top level subsets
        self.subsets = _checked_update(self.subsets, other.subsets)
        if self.main_splits is None:
            self.main_splits = other.main_splits
        elif other.main_splits is not None:
            self.main_splits.splits = _checked_update(self.main_splits.splits, other.main_splits.splits)

        self.languages |= other.languages

        return self


def _merge_metadata(ordered_metadata: list[tuple[int, SubMetadata]]) -> Metadata:
    """
    Merges metadata in order of specificity (highest first).

    Args:
        ordered_metadata: List of (specificity, SubMetadata) tuples.

    Returns:
        Merged Metadata instance.
    """

    metadata = Metadata()
    for _, sub_metadata in sorted(ordered_metadata, key=operator.itemgetter(0)):
        metadata = metadata.merge(sub_metadata)

    return metadata


@dataclass(slots=True)
class DataPipeState:
    """
    State of a data pipe.

    Attributes:
        dataset_name: Name of the dataset being processed.
        intermediate_directory: Directory for intermediate files.
        meld_directory: Directory for final MELD output.
        data_pipe_metadata: List of metadata configurations to apply.
        file_streams: Open file streams for processing.
        generated_files: Set of files generated during processing.
        splits: List of split configurations.
        metadata: Current dataset metadata.
        output_format: Format for output files.
        sentence_span_dir: Optional directory for sentence span files to
            segment document-level datasets.
    """

    dataset_name: str
    intermediate_directory: Path
    meld_directory: Path
    data_pipe_metadata: list[SubMetadata]
    file_streams: dict[Path, AbstractContextManager[Iterable[bytes]]] = field(default_factory=dict)
    generated_files: set[Path] = field(default_factory=set)
    splits: list[SplitData] = field(default_factory=list)
    metadata: _ProcessedDatasetMetadata = field(default_factory=_ProcessedDatasetMetadata)
    output_format: str = "parquet"
    sentence_span_dir: Path | None = None

    def scoped_state(self) -> Self:
        """
        Creates a new state with shared resources but without accumulated splits or metadata.

        Returns:
            New DataPipeState instance with shared file streams and
            generated files.
        """

        # State with shared file streams and generated files but without previous splits or metadata
        return self.__class__(
            self.dataset_name,
            self.intermediate_directory,
            self.meld_directory,
            self.data_pipe_metadata,
            self.file_streams,
            self.generated_files,
            output_format=self.output_format,
            sentence_span_dir=self.sentence_span_dir,
        )

    def update(self, other: Self) -> None:
        """
        Updates this state with data from another state.

        Args:
            other: The other DataPipeState to merge with.
        """

        self.splits.extend(other.splits)
        self.metadata.merge(other.metadata)

    def split_data_pipe_metadata(self, subset_hierarchy: list[str], split_name: str) -> Metadata:
        """
        Create metadata for a given subset hierarchy and split.

        Args:
            subset_hierarchy: Path through the subset hierarchy.
            split_name: Name of the split.

        Returns:
            The `Metadata` for the given split.
        """

        applicable_metadata = []

        for sub_metadata in self.data_pipe_metadata:
            # Treat empty selector as a general wildcard "*"
            if not sub_metadata.split:
                applicable_metadata.append(((0, 0), sub_metadata))
                continue

            specificity = max(
                (
                    selector.specificity()
                    for selector in sub_metadata.split
                    if selector.matches(subset_hierarchy, split_name)
                ),
                default=None,
            )
            if specificity is not None:
                applicable_metadata.append((specificity, sub_metadata))

        return _merge_metadata(applicable_metadata)

    def dataset_metadata(self) -> DatasetMetadata:
        """
        Constructs the final DatasetMetadata from the current state.

        Returns:
            Complete DatasetMetadata instance.
        """

        metadata = DatasetMetadata(
            self.dataset_name,
            self.metadata.subsets,
            self.metadata.languages,
            self.metadata.main_splits,
        )

        for subset in metadata._iter_subsets():
            for split_name, split in subset.splits.items():
                split.creation_metadata = self.split_data_pipe_metadata(subset.hierarchy, split_name)

        return metadata


@dataclass
class HuggingfaceModule:
    """
    Module for loading datasets from HuggingFace Hub.

    Attributes:
        config: HuggingfaceArguments configuration for dataset loading.
    """

    config: HuggingfaceArguments

    def _generate_splits_from(
        self,
        dataset: Dataset,
        target_split: str,
        subset_language: str,
        updated_subset: list[SubsetName],
    ) -> Iterator[SplitData]:
        """
        Generates a split from a dataset partition.

        Args:
            dataset: The dataset partition to create a split from.
            target_split: Name of the target split (train, validation,
                test).
            subset_language: Language of the subset.
            updated_subset: The subset hierarchy containing the
                resulting split.
        Yields:
            Generated Split instance.
        """

        yield SplitData(
            target_split,
            [DatasetSource(dataset, self.config.text_column, self.config.tag_column)],
            subset_language,
            updated_subset,
        )

    def _partition_by_language(
        self, dataset: Dataset, base_language: str | None
    ) -> Iterator[tuple[str, list[SubsetName], Dataset]]:
        """
        Partitions a dataset by language column.

        Args:
            dataset: Dataset to partition.
            base_language: Base language if no language column is
                configured.
        Yields:
            Tuples of (language, language_subset, data_subset).

        Raises:
            ValueError: If base_language is None and no language column
                is configured for the dataset.
        """

        if self.config.language_column is None:
            if base_language is None:
                raise ValueError(
                    "base_language needs to be specified when no language_column is configured for the dataset"
                )
            yield base_language, [], dataset
            return

        data_frame = dataset.to_polars()
        assert isinstance(data_frame, DataFrame), "DataSet did not return a single polars.DataFrame"

        language_column = data_frame[self.config.language_column]
        if language_column.n_unique() == 1:
            language = str(language_column.first())
            yield language, [], dataset
            return

        # NOTE: maintain_order is set to ensure deterministic data ordering
        for (language,), data_subset in data_frame.group_by(self.config.language_column, maintain_order=True):
            language = str(language)
            yield language, [SubsetName(language, "language")], Dataset.from_polars(data_subset)

    def _normalize_splits(self, dataset: DatasetDict, subset: str, single_subset: bool) -> Iterator[SplitData]:
        """
        Normalizes splits from a Huggingface dataset, handling language partitioning and split renaming.

        Args:
            dataset: DatasetDict containing the dataset partitions.
            subset: Name of the Huggingface subset.
            single_subset: Whether this is the only subset in the dataset.
        Yields:
            Normalized Split instances.

        Raises:
            ValueError: If a subset does not match the configured
                language_from_subset_name pattern.
            ValueError: If no language is configured for the dataset or
                a split and it cannot be inferred.
            ValueError: If a split does not match the configured
                split_naming_pattern.
        """

        # Rename all configured splits
        split_map = {
            name: target
            for name, target in zip(
                (self.config.train_name, self.config.validation_name, self.config.test_name),
                ("train", "validation", "test"),
            )
            if name is not None
        }

        subset_language_pattern = self.config.language_from_subset_name
        subset_is_language = subset_language_pattern is not None

        if subset_is_language:
            match = re.fullmatch(subset_language_pattern, subset)
            if match is None:
                raise ValueError(
                    f"Subset {subset!r} in {self.config.repo} does no match the language_from_subset_name pattern"
                )
            base_language = match.group("language")
        else:
            base_language = self.config.base_language

        # Treat the only subset of a single subset dataset as the "root" subset unless it contains language information
        huggingface_subset = (
            []
            if single_subset and not subset_is_language
            else [SubsetName(subset, "language" if subset_is_language else None)]
        )

        if self.config.split_naming_pattern is None:
            if base_language is None and self.config.language_column is None:
                raise ValueError("No language specified in the config and none can be inferred from the dataset")

            for original_split_name, split_name in split_map.items():
                for language, language_subset, dataset_partition in self._partition_by_language(
                    dataset[original_split_name], base_language
                ):
                    yield from self._generate_splits_from(
                        dataset_partition,
                        split_name,
                        language,
                        huggingface_subset + language_subset,
                    )
            return

        for original_split_name in dataset:
            match = re.fullmatch(self.config.split_naming_pattern, original_split_name)
            if match is None:
                raise ValueError(
                    f"Split {original_split_name!r} in {self.config.repo} does no match the split_naming_pattern"
                )
            split_components = match.groupdict()
            split = split_components.pop("split")

            subset_language = base_language
            current_subset = huggingface_subset.copy()
            if language := split_components.pop("language", None):
                current_subset.append(SubsetName(language, "language"))
                subset_language = language
            if year := split_components.pop("year", None):
                current_subset.append(SubsetName(year, "year"))

            if split_components:
                raise ValueError(f"Unknown subset type(s) in split_naming_pattern: {list(split_components)}")

            if subset_language is None:
                raise ValueError(
                    f"No language specified in the config and none can be inferred from the {original_split_name!r} subset"
                )

            for language, language_subset, dataset_partition in self._partition_by_language(
                dataset[original_split_name], subset_language
            ):
                yield from self._generate_splits_from(
                    dataset_partition,
                    split_map.get(split, split),
                    language,
                    current_subset + language_subset,
                )

    def _dataset_module(self, cache_dir: str) -> list[BuilderConfig] | None:
        """
        Retrieves dataset builder configurations from HuggingFace Hub.

        Args:
            cache_dir: Directory for caching dataset module files.

        Returns:
            List of BuilderConfig instances or `None` if no builder
            configs were found.
        """

        return dataset_module_factory(
            self.config.repo,
            revision=self.config.revision,
            cache_dir=cache_dir,
            trust_remote_code=self.config.trust_remote_code,
        ).builder_configs_parameters.builder_configs

    def _fast_subset_load(self, cache_dir: str) -> list[SplitData]:
        """
        Loads dataset subsets using fast parquet-based download and the low-level `datasets` API to construct `Dataset` instances from disk.

        Args:
            cache_dir: Directory for caching downloaded files.

        Returns:
            List of Split instances from loaded subsets.

        Raises:
            NotImplementedError: If the dataset contains data files that
                are not in the Parquet format.
            ValueError: If any loaded subset is not a DatasetDict.
            ValueError: If metadata config retrieval fails.
        """

        subset_configs = self._dataset_module(cache_dir)
        if subset_configs is None:
            raise ValueError("Failed to retrieve metadata configs for fast load from HuggingFace Hub")

        snapshot_dir = Path(
            snapshot_download(
                self.config.repo,
                repo_type="dataset",
                revision=self.config.revision,
                cache_dir=cache_dir,
                allow_patterns=[
                    file
                    for subset in subset_configs
                    for split in (subset.data_files.values() if subset.data_files is not None else [])
                    for file in split
                ],
            )
        )

        single_subset = len(subset_configs) == 1
        splits = []

        for subset_config in subset_configs:
            subset = subset_config.name
            if subset_config.data_files is None:
                logger.warning(f"No data files found for subset {subset!r}")
                continue

            if not isinstance(subset_config, ParquetConfig):
                raise NotImplementedError(
                    f"Fast subset loading is currently only implemented for parquet files but found config: {subset_config}"
                )

            dataset = load_dataset(
                "parquet",
                cache_dir=cache_dir,
                data_files={
                    split: str(snapshot_dir / path)
                    for split, paths in subset_config.data_files.items()
                    for path in paths
                },
            )
            if not isinstance(dataset, DatasetDict):
                raise TypeError(f"Expected DatasetDict but got {type(dataset).__name__}")

            splits.extend(self._normalize_splits(dataset, subset, single_subset))

        return splits

    def __call__(self, state: DataPipeState) -> DataPipeState:
        """
        Loads and normalizes dataset splits from HuggingFace Hub.

        Args:
            state: The DataPipeState to process.

        Returns:
            Updated DataPipeState with processed splits.

        Raises:
            ValueError: If the loaded dataset is not a DatasetDict.
        """

        if self.config.fast_subset_load:
            state.splits.extend(self._fast_subset_load(str(state.intermediate_directory / ".hf_cache")))
            return state

        cache_dir = str(state.intermediate_directory / ".hf_cache")
        subset_configs = self._dataset_module(cache_dir)
        extra_arguments = {}
        if subset_configs is not None:
            # Avoid pandas backend interpreting tokens like "None" as null values for CSV/TSV datasets
            for config in subset_configs:
                if isinstance(config, CsvConfig):
                    extra_arguments[config.name] = {
                        "keep_default_na": False,
                    }

        if isinstance(self.config.data_files, SubsetDataFiles):
            subsets = self.config.data_files.subsets.keys()
            file_subsets = True
        else:
            subsets = get_dataset_config_names(self.config.repo, revision=self.config.revision)
            file_subsets = False

        single_subset = len(subsets) == 1
        for subset in subsets:
            if isinstance(self.config.data_files, SubsetDataFiles):
                data_files = self.config.data_files.subsets[subset]
            else:
                data_files = self.config.data_files

            if not single_subset and data_files is not None:
                # Use separate local cache directories to avoid any potential issues when splitting
                # a dataset into subsets using only data_files
                current_cache_dir = f"{cache_dir}_{subset}"
            else:
                current_cache_dir = cache_dir

            dataset = load_dataset(
                self.config.repo,
                "default" if file_subsets else subset,
                revision=self.config.revision,
                cache_dir=current_cache_dir,
                trust_remote_code=self.config.trust_remote_code,
                data_files=data_files,
                **extra_arguments.get(subset, {}),
            )
            if not isinstance(dataset, DatasetDict):
                raise TypeError(f"Expected DatasetDict or IterableDatasetDict but got {type(dataset).__name__}")

            state.splits.extend(self._normalize_splits(dataset, subset, single_subset))

        return state


@dataclass
class GitModule:
    """
    Module for cloning and extracting files from Git repositories.

    Attributes:
        config: GitLoaderArguments or GitStep configuration for
            repository cloning and file extraction.
    """

    config: GitLoaderArguments | GitStep

    def __call__(self, state: DataPipeState) -> DataPipeState:
        """
        Clones the Git repository and extracts specified files.

        Args:
            state: The DataPipeState to process.

        Returns:
            Updated DataPipeState with generated files.
        """

        if isinstance(self.config, GitStep):
            directory = Path(self.config.directory)
            file_spec = [directory / file for file in self.config.files]
        else:
            # Collect all files from the subset definitions
            file_spec = []
            for subset in self.config.subsets.values():
                directory = Path(subset.directory)
                file_spec.extend(
                    directory / file_glob
                    for file_globs in itertools.chain(subset.train, subset.validation, subset.test)
                    for file_glob in (
                        (file_globs,) if isinstance(file_globs, str) else itertools.chain(*file_globs.values())
                    )
                )

        # Hash the repo, revision and file_spec to track a specific configuration
        manifest_hash = hashlib.sha256(
            (self.config.repo + self.config.revision + "\n".join(map(str, file_spec))).encode("utf-8")
        ).hexdigest()
        git_manifest_path = state.intermediate_directory / f".git_manifest_{manifest_hash}.txt"

        # Skip cloning the repository if it has already been processed once with the same revision and file_spec
        # Assumes the previous clone and archive calls have been successful in this case since the manifest is only written at the end of this module
        has_manifest = git_manifest_path.exists()
        if not has_manifest:
            # Collect all required files from the Git repository which will be cloned first if necessary
            download.git_download(
                self.config.repo, self.config.revision, state.intermediate_directory, file_spec, self.config.keep_repo
            )

        # Collect all archived paths, including those matched by globs
        generated_files = {
            file for file_glob in file_spec for file in state.intermediate_directory.glob(str(file_glob))
        }

        # Store paths to the files extracted from the git repository
        with git_manifest_path.open("w") as file:
            for path in sorted(generated_files):
                file.write(f"{path}\n")

        state.generated_files.update(generated_files)
        return state


class ChecksumValidationError(Exception):
    """Raised when a checksum differs from an expected value."""


def _verify_sha256(file_path: Path, checksum: str) -> None:
    """
    Verifies a file's SHA-256 checksum.

    Args:
        file_path: Path to the file to verify.
        checksum: Expected SHA-256 checksum.

    Raises:
        ChecksumValidationError: If the checksum does not match.
    """

    with file_path.open("rb", buffering=0) as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
        if actual != checksum:
            raise ChecksumValidationError(
                f'Checksum validation failed for file "{file_path}", Sha-256 {checksum!r} != {actual!r}'
            )


@dataclass(slots=True)
class DownloadModule:
    """
    Module for downloading files from URLs with SHA-256 verification.

    Attributes:
        config: DownloadStep configuration containing URLs and
            checksums.
    """

    config: DownloadStep

    def __call__(self, state: DataPipeState) -> DataPipeState:
        """
        Downloads files from configured URLs and verifies their SHA-256 checksums.

        Args:
            state: The DataPipeState to process.

        Returns:
            Updated DataPipeState with downloaded files added to
            generated_files.
        """

        for url in self.config.urls:
            path = Path(urlparse(url.url).path)
            target_path = state.intermediate_directory / path.name
            # Skip any further checks and downloading of the file other than checksum verification
            # when it already exists in the target directory with the name that could be parsed from the URL
            if target_path.is_file():
                result = target_path
            else:
                state.intermediate_directory.mkdir(exist_ok=True, parents=True)
                # Skip downloading if the file already exists and the filename could not already be correctly inferred from the URL
                result = download.download(
                    url.url, state.intermediate_directory, skip_if_file_exists=True, stream=False
                )

            # Verify checksum of the downloaded file
            _verify_sha256(result, url.sha256)

            state.generated_files.add(result)

        return state


@dataclass(slots=True)
class GoogleDocsModule:
    """
    Module for downloading files from Google Docs links with SHA-256 verification.

    Attributes:
        config: GoogleDocsStep configuration containing Google Docs URLs
            and metadata.
    """

    config: GoogleDocsStep

    def __call__(self, state: DataPipeState) -> DataPipeState:
        """
        Downloads files from configured Google Docs links and verifies their SHA-256 checksums.

        Args:
            state: The DataPipeState to process.

        Returns:
            Updated DataPipeState with downloaded files.
        """

        state.intermediate_directory.mkdir(exist_ok=True, parents=True)

        for docs_link in self.config.urls:
            target_path = state.intermediate_directory / docs_link.target_filename
            # Skip downloading when the file already exists in the target directory
            if not target_path.is_file():
                # Uses expanduser since gdown does not handle paths starting with ~ otherwise
                gdown.download(docs_link.url, str(target_path.expanduser()))

            # Verify checksum of the downloaded file
            _verify_sha256(target_path, docs_link.sha256)

            state.generated_files.add(target_path)

        return state


@dataclass(slots=True)
class ExtractModule:
    """
    Module for extracting files from archives (zip, tar, etc.).

    Attributes:
        config: ExtractStep configuration specifying source archive and
            extracted files.
    """

    config: ExtractStep

    def __call__(self, state: DataPipeState) -> DataPipeState:
        """
        Extracts and tracks files from a configured archive.

        Args:
            state: The DataPipeState to process.

        Returns:
            Updated DataPipeState with extracted files.
        """

        source_file = state.intermediate_directory / self.config.from_file

        with (
            stream
            if (stream := state.file_streams.pop(source_file, None)) is not None
            else contextlib.nullcontext(source_file) as file
        ):
            state.generated_files.update(
                extract(
                    file,
                    state.intermediate_directory,
                    self.config.files,
                    "".join(source_file.suffixes),
                    member_globs=self.config.use_globs,
                )
            )

        return state


type _SplitReaderFunction = Callable[
    [Path, str | dict[str, str], FileSource | dict[str, FileSource]], dict[str, SplitPaths]
]


class SplitReaderRegistry(Registry[_SplitReaderFunction]):
    """Registry for CoNLL preprocessor functions."""


@SplitReaderRegistry.register("legalnero_standoff_split_columns")
def _read_legalnero_standoff_split_columns(
    intermediate_path: Path, base_directories: str | dict[str, str], source: FileSource | dict[str, FileSource]
) -> dict[str, SplitPaths]:
    """
    Reads LegalNER standoff split configurations from tab-separated files.

    Args:
        intermediate_directory: Directory for intermediate files used by the Data Pipe (unused).
        base_directories: Dictionary with "annotations" and "text" base
            paths.
        source: Single file source containing tab-separated path and
            split pairs.

    Returns:
        Mapping of split names to lists of standoff file paths.

    Raises:
        ValueError: If the source is not a single file.
        ValueError: If base_directories is missing "annotations" or
            "text" paths.
    """

    if isinstance(source, dict):
        raise TypeError(f"Expected a single file containing split metadata, got {source!r}")

    if (
        isinstance(base_directories, str)
        or (text_base := base_directories.get("text")) is None
        or (annotation_base := base_directories.get("annotations")) is None
    ):
        raise ValueError(
            'standoff_path_split_columns reader requires a dictionary of base paths for "annotations" and "text"'
        )

    splits = defaultdict(list)

    text_base = Path(text_base)
    annotation_base = Path(annotation_base)

    for line in source.lines():
        path, split = line.strip().split("\t")
        path = Path(path)
        splits[split].append(
            {
                "annotations": [annotation_base / path.with_suffix(".ann")],
                "text": [text_base / path.with_suffix(".txt")],
            }
        )

    return dict(splits)


@SplitReaderRegistry.register("agriner_standoff_split_json")
def _read_agriner_standoff_split_json(
    intermediate_path: Path, base_directories: str | dict[str, str], split_files: FileSource | dict[str, FileSource]
) -> dict[str, SplitPaths]:
    """
    Reads AgriNER standoff split configurations from JSON files.

    Args:
        intermediate_directory: Directory for intermediate files used by the Data Pipe (unused).
        base_directories: Dictionary with "annotations" and "text" base
            paths.
        split_files: JSON files of a split containing the file path of
            their standoff source document.

    Returns:
        Mapping of split names to lists of standoff file paths.

    Raises:
        ValueError: If split_files is a single file rather than one file
            per split.
    """

    if isinstance(split_files, FileSource):
        raise TypeError(f"One path per split expected, got {split_files.path!r}")

    split_paths = {}

    for split, source in split_files.items():
        standoff_paths = []
        if isinstance(base_directories, str):
            directory = Path(base_directories)
        else:
            directory = Path(base_directories[split])

        for document in json.loads(source.read()):
            file_path = Path(document["file"])
            standoff_paths.append(
                {"annotations": [directory / file_path.with_suffix(".ann")], "text": [directory / file_path]}
            )

        split_paths[split] = standoff_paths

    return split_paths


def _split_files_to_line_sources(
    intermediate_directory: Path, split_files: str | dict[str, str]
) -> FileSource | dict[str, FileSource]:
    """
    Converts split file paths to FileSource instances for reading.

    Args:
        intermediate_directory: Directory containing the split files.
        split_files: Path or mapping from split names to file paths.

    Returns:
        FileSource or mapping of FileSource instances for each path if
        split_files is a dictionary.
    """

    if isinstance(split_files, str):
        return FileSource(intermediate_directory / split_files)

    return {split: FileSource(intermediate_directory / file) for split, file in split_files.items()}


@SplitReaderRegistry.register("somesci_standoff_split_json")
def _read_somesci_standoff_split_json(
    intermediate_directory: Path, base_directories: str | dict[str, str], source: FileSource | dict[str, FileSource]
) -> dict[str, SplitPaths]:
    """
    Reads SomeSci standoff split configurations from JSON files.

    Args:
        intermediate_directory: Directory for intermediate files used by the Data Pipe.
        base_directories: Base directory paths for standoff files,
            either as a string or per-split mapping.
        source: Single file containing split definitions as JSON.

    Returns:
        Dictionary mapping split names to lists of standoff file paths.

    Raises:
        ValueError: If source is not a single file.
    """

    if isinstance(source, dict):
        raise TypeError(f"Expected a single file containing split metadata, got {source!r}")

    split_paths = {}

    for split, ids in json.loads(source.read()).items():
        standoff_paths = []
        if isinstance(base_directories, str):
            directory = Path(base_directories)
        else:
            directory = Path(base_directories[split])

        for document_id in ids:
            text_path = directory / f"{document_id}.txt"
            if (intermediate_directory / text_path).exists():
                standoff_paths.append({"annotations": [text_path.with_suffix(".ann")], "text": [text_path]})

        split_paths[split] = standoff_paths

    # Filter empty splits
    return {split: files for split, files in split_paths.items() if files}


class _ValidateSplitsMixin:
    """
    Mixin for validating split file paths and ensuring consistent ordering.

    Attributes:
        available_splits: List of (split name, SplitPaths) tuples to
            validate.
    """

    available_splits: list[tuple[str, SplitPaths]]

    def sort_splits(self) -> None:
        """
        Sorts all split file paths in place to ensure deterministic and consistent ordering.

        Sorts paths within each split and orders the splits themselves for reproducibility.
        """

        for split_index, (split_name, paths) in enumerate(self.available_splits):
            for shard_index, path in enumerate(paths):
                if isinstance(path, Path):
                    continue

                # Sort values, then sort the full dictionary by key to ensure consistency
                for name, subpaths in path.items():
                    path[name] = sorted(subpaths)

                paths[shard_index] = dict(sorted(path.items(), key=operator.itemgetter(0)))

            self.available_splits[split_index] = (split_name, sorted(paths, key=_shard_sort_key))

    def validate(self, base_directory: Path) -> list[Path]:
        """
        Validates that all split file paths exist. In the case of globs,
        this function validates that they match at least one path.

        Args:
            base_directory: Base directory to resolve relative paths
                against.

        Returns:
            List of all missing or unreadable file paths.
        """

        return [
            base_directory / file_glob
            for _, paths in self.available_splits
            for file_globs in paths
            for file_glob in ((file_globs,) if isinstance(file_globs, Path) else itertools.chain(*file_globs.values()))
            # Returns true once there is at least one path that matches the glob
            if not any(True for _ in base_directory.glob(str(file_glob)))
        ]


@dataclass
class ReadSplitModule(_ValidateSplitsMixin):
    """
    Module for reading dataset splits from files, indicating which files belong to which split.

    Attributes:
        config: ReadSplitStep configuration options.
        subset: List with a single subset if a subset is configured.
        available_splits: List of (split name, SplitPaths) tuples.
    """

    config: ReadSplitStep

    def __post_init__(self) -> None:
        """
        Initialize ReadSplitModule with configuration.
        """
        self.subset = [] if self.config.subset is None else [SubsetName(self.config.subset)]

    def __call__(self, state: DataPipeState) -> DataPipeState:
        """
        Reads dataset splits using the configured reader.

        Args:
            state: The DataPipeState to process.

        Returns:
            Updated DataPipeState with parsed splits.

        Raises:
            DataPipeError: If splits fail to validate due to missing or
                unreadable files.
        """

        reader = SplitReaderRegistry.get(self.config.splits_reader)
        split_name_map = self.config.split_name_map

        self.available_splits = [
            (name if split_name_map is None else split_name_map[name], files)
            for name, files in reader(
                state.intermediate_directory,
                self.config.directories,
                _split_files_to_line_sources(state.intermediate_directory, self.config.split_files),
            ).items()
        ]

        self.sort_splits()

        if missing := self.validate(state.intermediate_directory):
            raise DataPipeError(f"Splits failed to validate. Missing or unreadable files: {missing}")

        state.splits.extend(
            SplitData(
                name,
                _create_sources(state.intermediate_directory, files),
                self.config.language,
                self.subset,
                self.config.metadata,
            )
            for name, files in self.available_splits
        )
        return state


def _add_paths_to_directory(directory: Path, sub_paths: SplitFiles) -> SplitPaths:
    """
    Prepend a base directory path to all sub-paths.

    Args:
        directory: The base directory path to prepend.
        sub_paths: Sub-paths which may be strings or mappings of file
            types to paths.

    Returns:
        List of paths with directory prepended.
    """

    return [
        directory / sub_path
        if isinstance(sub_path, str)
        else {file_type: [directory / path for path in paths] for file_type, paths in sub_path.items()}
        for sub_path in sub_paths
    ]


def _create_sources(intermediate_directory: Path, files: SplitPaths) -> list[DataSource]:
    """
    Create DataSource instances from file specifications. If `files` contain globs,
    matched file paths will be sorted for reproducibility.

    Args:
        intermediate_directory: Directory containing the files.
        files: File specifications as paths or glob patterns.

    Returns:
        List of DataSource instances.
    """

    sources = []
    for glob_or_mapping in files:
        if isinstance(glob_or_mapping, Path):
            sources.extend(FileSource(file) for file in sorted(intermediate_directory.glob(str(glob_or_mapping))))
            continue

        sources.append(
            {
                file_type: [
                    FileSource(file) for glob in globs for file in sorted(intermediate_directory.glob(str(glob)))
                ]
                for file_type, globs in glob_or_mapping.items()
            }
        )

    return sources


_ShardTypeAdapter = TypeAdapter(dict[str, list[Path]])


def _shard_sort_key(shard: Path | dict[str, list[Path]]) -> str:
    """
    Generate a sort key for shards to ensure consistent ordering.

    Args:
        shard: A Path or dictionary mapping file types to paths.

    Returns:
        String key for sorting shards.
    """

    if isinstance(shard, Path):
        return str(shard)

    # Assumes dictionary shards are already completely internally sorted
    # Serializes the full dictionary to avoid issues in cases where the first path of the first key are shared by all shards
    return _ShardTypeAdapter.dump_json(shard).decode()


@dataclass
class SplitModule(_ValidateSplitsMixin):
    """
    Data pipe module for splitting datasets into train, validation and test sets.

    Attributes:
        config: SplitStep configuration.
        train: List of training file paths.
        validation: List of validation file paths.
        test: List of test file paths.
        subset: List with a single subset if a subset is configured.
        available_splits: List of (split name, SplitPaths) tuples.
    """

    config: SplitStep

    def __post_init__(self) -> None:
        """
        Initialize SplitModule with configuration.
        """
        directory = Path(self.config.directory)

        self.train = _add_paths_to_directory(directory, self.config.train or [])
        self.validation = _add_paths_to_directory(directory, self.config.validation or [])
        self.test = _add_paths_to_directory(directory, self.config.test or [])
        self.subset = [] if self.config.subset is None else [SubsetName(self.config.subset)]
        self.available_splits = [
            (name, files)
            for name, files in (("train", self.train), ("validation", self.validation), ("test", self.test))
            if files
        ]

    def __call__(self, state: DataPipeState) -> DataPipeState:
        """
        Reads split file paths from configured directories.

        Args:
            state: The DataPipeState to process.

        Returns:
            Updated DataPipeState with parsed splits.

        Raises:
            DataPipeError: If splits fail to validate due to missing or
                unreadable files.
        """

        self.sort_splits()
        if missing := self.validate(state.intermediate_directory):
            raise DataPipeError(f"Splits failed to validate. Missing or unreadable files: {missing}")

        state.splits.extend(
            SplitData(
                name,
                _create_sources(state.intermediate_directory, files),
                self.config.language,
                self.subset,
                self.config.metadata,
            )
            for name, files in self.available_splits
        )
        return state


@dataclass(slots=True)
class _LabelSet:
    """
    Keeps track of unique labels and BIO tags from NER documents.

    Attributes:
        labels: Mapping of tagset names to sets of entity labels.
        bio_labels: Mapping of tagset names to sets of BIO tags.
        tagsets_to_infer: Set of tagset names for which to infer labels
            from data.
    """

    labels: dict[str, set[str]]
    bio_labels: dict[str, set[str]]
    tagsets_to_infer: set[str]

    def add_labels(self, document: NERDocument) -> None:
        """
        Extracts labels from a document's annotations and adds them to the tracked label sets.

        Args:
            document: NERDocument from which labels are extracted.
        """

        if document.bio is not None:
            for sequence in document.bio:
                for tagset, tags in sequence.labels.items():
                    if tagset not in self.tagsets_to_infer:
                        continue
                    self.bio_labels[tagset].update(map(str, tags))

        for sequence in document.spans:
            for tagset, annotations in sequence.labels.items():
                if tagset not in self.tagsets_to_infer:
                    continue
                self.labels[tagset].update(annotation.label for annotation in annotations)


def _annotation_sort_key(annotation: Annotation) -> tuple[int, int]:
    """
    Generates a sort key for annotations based on their first and last span positions.

    Args:
        annotation: Annotation to generate a sort key for.

    Returns:
        Tuple of (start position, end position) for sorting annotations.
    """

    # Treat discontinuous spans as a single, long span for sorting
    return annotation.spans[0].start, annotation.spans[-1].stop


def _sort_annotations(document: NERDocument) -> None:
    """
    Sorts annotations within each labeled text span in a document in-place for consistency and reproducibility.

    Args:
        document: NERDocument with annotations to sort.
    """

    for labeled_text in document.spans:
        labeled_text.labels = {
            tagset: sorted(labels, key=_annotation_sort_key) for tagset, labels in labeled_text.labels.items()
        }


@runtime_checkable
class _ReaderWithTagset(Protocol):
    """Runtime checkable protocol for identifying readers with (optional) configured tagsets."""

    tagsets: list[NamedTagSet] | None


@dataclass
class ConvertModule:
    """
    Module for converting dataset files to the MELD parquet format.

    Attributes:
        config: ConvertStep or DatasetConvertStep configuration for the
            conversion process.
    """

    config: ConvertStep | DatasetConvertStep

    def __post_init__(self) -> None:
        """
        Initializes the ConvertModule with a reader based on the configuration.
        """
        if isinstance(self.config, DatasetConvertStep):
            self._reader = DatasetReader(self.config)
        else:
            self._reader = reader_from_config(self.config.reader)

    def _label_set(self) -> _LabelSet:
        """
        Creates a label set configured based on the reader's tagsets.

        Returns:
            _LabelSet with label sets for label inference or explicit
            label maps.
        """

        # Skip inference of possible labels in the dataset if a label_map has been explicitly defined
        if isinstance(self._reader, _ReaderWithTagset) and self._reader.tagsets is not None:
            # Only infer labels for tagsets without an explicit label map
            infer_labels = {tagset.name for tagset in self._reader.tagsets if tagset.label_map is None}
            bio_labels = {
                tagset.name: set() if tagset.label_map is None else set(map(str, tagset.label_map.values()))
                for tagset in self._reader.tagsets
            }

            labels = {
                tagset.name: set()
                if tagset.label_map is None
                else {label.entity_type for label in tagset.label_map.values() if label.entity_type is not None}
                for tagset in self._reader.tagsets
            }
            return _LabelSet(labels, bio_labels, infer_labels)

        # Assumes other readers only support a single, default tagset
        return _LabelSet({DEFAULT_TAGSET: set()}, {DEFAULT_TAGSET: set()}, {DEFAULT_TAGSET})

    def __call__(self, state: DataPipeState) -> DataPipeState:
        """
        Converts all splits to the MELD parquet format, infers tag sets if necessary, and writes output files.

        Args:
            state: The DataPipeState containing splits to convert.

        Returns:
            Updated DataPipeState with metadata of the processed splits.

        Raises:
            ValueError: If an empty document is encountered during
                processing.
            ValueError: If overlapping subsets are encountered.
            ValueError: If an empty split is encountered.
            ValueError: If the output format is unsupported.
            ValueError: If reading a split fails.
        """

        if not state.splits:
            raise ValueError("ConvertModule received no data splits to process")

        # NOTE: Could support additional output formats in the future
        if state.output_format != "parquet":
            raise ValueError(f"Unsupported output format: {state.output_format}")

        main_splits = None
        subsets = {}
        languages = set()
        label_sets: dict[str, _LabelSet] = {}

        state.meld_directory.mkdir(exist_ok=True)

        for split in tqdm(state.splits, unit=" splits", desc="Converting splits"):
            # Normalize split language
            try:
                language_tag = Language.get(split.language)
                split.language = language_tag.to_alpha3() + (
                    "" if language_tag.script is None else ("-" + language_tag.script)
                )
            except LanguageTagError:
                logger.warning(f"Keeping invalid language code {split.language!r}")

            languages.add(split.language)

            # Filename with format based extension
            split_filename = f"{split.name}.{state.output_format}"
            subset_slug = ""
            if not split.subset:
                path = state.meld_directory / split_filename
                if main_splits is None:
                    main_splits = SubsetMetadata._default_with_language(split.language)
                current_subset_metadata = main_splits
                current_split_metadata = main_splits.splits[split.name] = SplitMetadata(
                    path.relative_to(state.meld_directory)
                )
            else:
                subset_slug = split.slug()

                path = state.meld_directory / subset_slug
                path.mkdir(exist_ok=True)
                path /= split_filename

                component_dict: SubsetHierarchy = subsets
                for component in split.subset[:-1]:
                    name = component.name
                    if name not in component_dict:
                        component_dict[name] = {}

                    new_dict = component_dict[name]
                    if not isinstance(new_dict, dict):
                        raise TypeError(f"Overlapping subsets encountered: {new_dict}")

                    component_dict = new_dict

                last_subset = split.subset[-1].name
                if (current_subset_metadata := component_dict.get(last_subset)) is None:
                    current_subset_metadata = SubsetMetadata._default_with_language(split.language)
                    component_dict[last_subset] = current_subset_metadata
                elif not isinstance(current_subset_metadata, SubsetMetadata):
                    raise TypeError(f"Overlapping subsets encountered: {last_subset}")

                current_split_metadata = current_subset_metadata.splits[split.name] = SplitMetadata(
                    path.relative_to(state.meld_directory)
                )

            documents = iter(self._reader.read_split(split))
            # Infers whether documents are pre-tokenized from the first document
            try:
                first_document = next(documents)
                current_subset_metadata.pre_tokenized = has_tokens = first_document.bio is not None
                current_subset_metadata.tagsets = tagsets = sorted(first_document.spans[0].labels.keys())
            except StopIteration:
                raise ValueError(f"Encountered empty split {split}")

            if (subset_labels := label_sets.get(subset_slug)) is None:
                label_sets[subset_slug] = subset_labels = self._label_set()

            sentence_span_filename = f"{state.dataset_name}--{subset_slug}--{split.name}.parquet"
            if state.sentence_span_dir is None:
                sentence_span_file = resources.files("meld") / "package_data/sentence_spans" / sentence_span_filename
            else:
                sentence_span_file = state.sentence_span_dir / sentence_span_filename

            # Only sentence tokenize if the dataset isn't already segmented into sentences
            sentence_boundaries = state.split_data_pipe_metadata(
                [subset.name for subset in split.subset], split.name
            ).sentence_boundaries

            splitter = (
                contextlib.nullcontext()
                if sentence_boundaries == "full"
                else SentenceSplitter(
                    sentence_boundaries,
                    sentence_span_file,
                    read_spans=not state.sentence_span_dir,
                )
            )

            document_count = 0
            sequence_count = 0
            filtered_count = 0

            with (
                NERParquetWriter.open(
                    state.dataset_name,
                    subset_slug,
                    split.name,
                    path,
                    has_tokens,
                    tagsets,
                ) as writer,
                splitter as splitter,
            ):
                for document in itertools.chain((first_document,), documents):
                    # Sort annotations for reproducibility first
                    _sort_annotations(document)
                    if splitter is not None:
                        document = splitter.sentence_tokenize(document)

                    # Sanity check
                    if any(not span.text for span in document.spans):
                        if self.config.filter_empty_documents:
                            filtered_count += 1
                            continue
                        raise ValueError(f"Empty text span found in {document!r}")

                    if subset_labels.tagsets_to_infer:
                        subset_labels.add_labels(document)

                    writer.write_document(document)
                    document_count += 1
                    sequence_count += len(document.spans)

            if filtered_count > 0:
                logger.warning(f"Removed {filtered_count} empty documents in split {subset_slug}.{split.name}")

            current_split_metadata.document_count = document_count
            current_split_metadata.sequence_count = sequence_count
            # Sorts labels for consistency and sets "bio_labels" to None if no BIO labels were found for any tagset
            current_subset_metadata.labels = {
                name: sorted(tagset_labels) for name, tagset_labels in subset_labels.labels.items()
            }
            current_subset_metadata.bio_labels = (
                None
                if not any(subset_labels.bio_labels.values())
                else {name: sorted(tagset_labels) for name, tagset_labels in subset_labels.bio_labels.items()}
            )

        state.metadata = _ProcessedDatasetMetadata(
            subsets,
            main_splits,
            languages,
        )
        return state


type DataPipeModule = Callable[[DataPipeState], DataPipeState]


class DataPipeError(Exception):
    """Exception raised for errors in a data pipe."""


@dataclass(slots=True)
class MetadataStep:
    """
    Step marker for the `MetadataModule`.

    Attributes:
        step: Literal "metadata" indicating the step type.
    """

    step: Literal["metadata"] = "metadata"


@dataclass(slots=True)
class MetadataModule:
    """
    Module for writing dataset metadata to disk after data processing is complete.
    """

    def __call__(self, state: DataPipeState) -> DataPipeState:
        """
        Writes dataset metadata to the output directory.

        Args:
            state: The DataPipeState containing metadata to write.

        Returns:
            Unchanged DataPipeState.
        """

        metadata = state.dataset_metadata()
        metadata.dump(state.meld_directory)

        return state


def _config_to_stage(
    config: DataPipeStep | MetadataStep, data_pipe_metadata: list[SubMetadata], initial_steps: set[str]
) -> DataPipeModule:
    """
    Factory function for initializing a module from its corresponding DataPipeStep configuration.

    Args:
        config: DataPipeStep or MetadataStep configuration.
        data_pipe_metadata: List of sub-metadata for the data pipe.
        initial_steps: Set of step IDs considered already completed.

    Returns:
        Initialized DataPipeModule instance.
    """

    match config:
        case DownloadStep():
            return DownloadModule(config)
        case GoogleDocsStep():
            return GoogleDocsModule(config)
        case GitStep():
            return GitModule(config)
        case ExtractStep():
            return ExtractModule(config)
        case SplitStep():
            return SplitModule(config)
        case ReadSplitStep():
            return ReadSplitModule(config)
        case ConvertStep() | DatasetConvertStep():
            return ConvertModule(config)
        case NestedDataPipeStep():
            return DataPipe(config.data_pipe, data_pipe_metadata, assume_completed=initial_steps)
        case MetadataStep():
            return MetadataModule()


_STEP_DEPENDENCIES = {
    "download": None,
    "google_docs": None,
    "data_pipe": None,
    "git": None,
    "extract": ["download", "google_docs", "git", "huggingface"],
    "splits": ["download", "google_docs", "extract", "git", "huggingface"],
    "read_splits": ["download", "google_docs", "extract", "git", "huggingface"],
    "convert": ["splits", "read_splits", "git", "huggingface"],
    "columns": ["huggingface"],
    "metadata": ["data_pipe", "convert", "columns"],
}

_DUPLICATES_ALLOWED = {"download", "git", "google_docs", "splits", "read_splits", "extract", "data_pipe"}


class DataPipe:
    """
    Orchestrates dataset processing through a series of modular steps.

    Manages the execution order of data pipe steps, validates dependencies,
    and maintains state throughout the pipeline.

    Args:
        step_definitions: Iterable of DataPipeStep and MetadataStep
            configurations.
        predefined_dataset_metadata: List of sub-metadata providing
            additional information about the dataset to process.
        loader: Optional loader module to initialize the first
            stage.
        assume_completed: Set of step IDs to consider already
            completed for validation purposes in nested data pipes.

    Raises:
        DataPipeError: If a duplicate step is encountered of a type
            for which duplicates are not allowed.
        DataPipeError: If a step's dependencies are not satisfied.
    """

    _stages: list[list[DataPipeModule]]

    def __init__(
        self,
        step_definitions: Iterable[DataPipeStep | MetadataStep],
        predefined_dataset_metadata: list[SubMetadata] | None = None,
        loader: Loader | None = None,
        assume_completed: Iterable[str] | None = None,
    ) -> None:
        if predefined_dataset_metadata is None:
            predefined_dataset_metadata = []
        self._predefined_dataset_metadata = predefined_dataset_metadata

        self._stage_definitions = []
        completed_steps = set() if loader is None else {loader.loader}
        if assume_completed:
            completed_steps.update(assume_completed)
        initial_steps = completed_steps.copy()
        last_step = loader

        for i, step in enumerate(step_definitions):
            step_type = step.step
            if step_type in completed_steps and step_type not in _DUPLICATES_ALLOWED:
                raise DataPipeError(f"Duplicate {step_type!r} step at position {i}")
            if step_type == last_step:
                self._stage_definitions[-1].append(step)
                continue
            dependencies = _STEP_DEPENDENCIES[step_type]
            if not (dependencies is None or any(dependency in completed_steps for dependency in dependencies)):
                raise DataPipeError(
                    f"Missing dependency for {step_type!r} step at position {i}."
                    f" Needs at least one completed step of {_STEP_DEPENDENCIES[step_type]}"
                )
            self._stage_definitions.append([step])
            completed_steps.add(step_type)
            last_step = step_type

        match loader:
            case HuggingfaceLoader(arguments=arguments):
                self._stages = [[HuggingfaceModule(arguments)]]
            case GitLoader(arguments=arguments):
                self._stages = [[GitModule(arguments)]]
            case None | GenericLoader():
                self._stages = []

        self._stages.extend(
            [_config_to_stage(step, predefined_dataset_metadata, initial_steps) for step in parallel_steps]
            for parallel_steps in self._stage_definitions
        )

    def __call__(self, state: DataPipeState) -> DataPipeState:
        """
        Executes the data pipe with the given state.

        Args:
            state: Initial DataPipeState to process.

        Returns:
            Updated DataPipeState after all stages have run.
        """

        # Create a scoped state to support nested data pipes with their own contexts
        sub_state = state.scoped_state()
        for stage in self._stages:
            for step in stage:
                sub_state = step(sub_state)

        # Merge changes into the shared state
        state.update(sub_state)
        return state

    def run(
        self,
        dataset_name: str,
        intermediate_directory: Path,
        meld_directory: Path,
        sentence_span_dir: Path | None = None,
    ) -> DataPipeState:
        """
        Creates a new DataPipeState and executes the data pipe.

        Args:
            dataset_name: Name of the dataset being processed.
            intermediate_directory: Directory for intermediate
                downloaded, extracted or cached files.
            meld_directory: Directory for final MELD output.
            sentence_span_dir: Optional directory for sentence span
                files used for applying sentence segmentation to
                document-level datasets.

        Returns:
            Final DataPipeState after data pipe execution.
        """

        return self(
            DataPipeState(
                dataset_name,
                intermediate_directory,
                meld_directory,
                self._predefined_dataset_metadata,
                sentence_span_dir=sentence_span_dir,
            )
        )


def dataset_pipe(dataset: MELDDataset) -> DataPipe:
    """
    Constructs a DataPipe from a dataset configuration.

    Args:
        dataset: MELDDataset configuration from which to construct a
            DataPipe.

    Returns:
        Configured DataPipe instance.

    Raises:
        DataPipeError: If a subset language is not defined in a
            GitLoader configuration and no base language is configured.
    """

    loader = dataset.source
    match loader:
        case HuggingfaceLoader(arguments=arguments):
            return DataPipe(arguments.data_pipe + [MetadataStep()], dataset.metadata, loader)
        case GitLoader(arguments=arguments):
            base_language = loader.arguments.base_language
            full_data_pipe = []
            for subset, subset_arguments in arguments.subsets.items():
                language = subset_arguments.language or base_language
                if language is None:
                    raise DataPipeError(f"Missing subset language or base_language for subset {subset!r}")
                # Turn subset definition into split steps for each nested data pipe
                full_data_pipe.append(
                    NestedDataPipeStep(
                        "data_pipe",
                        [
                            SplitStep(
                                "splits",
                                language,
                                directory=subset_arguments.directory,
                                train=subset_arguments.train,
                                validation=subset_arguments.validation,
                                test=subset_arguments.test,
                                # Handle __root__ key to indicate a default subset
                                subset=subset if subset != "__root__" else None,
                            )
                        ]
                        + (subset_arguments.data_pipe or arguments.default_data_pipe),
                    )
                )

            full_data_pipe.append(MetadataStep())

            return DataPipe(full_data_pipe, dataset.metadata, loader)
        case GenericLoader(arguments=arguments):
            return DataPipe(arguments.download_data_pipe + [MetadataStep()], dataset.metadata)
