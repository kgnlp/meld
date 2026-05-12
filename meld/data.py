"""Main entrypoint for data download and management."""

import csv
import dataclasses
import json
import logging
import os
import sys
from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from importlib import resources
from json import JSONDecodeError
from pathlib import Path
from typing import IO

import bibtexparser
import polars as pl
import pyarrow as pa
from huggingface_hub import DatasetCard, snapshot_download
from pyarrow.parquet import ParquetWriter
from pydantic import TypeAdapter, ValidationError
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from meld import data_stats, formats
from meld.data_pipes import dataset_pipe
from meld.formats import (
    DEFAULT_TAGSET,
    METADATA_FILENAME,
    PROCESSED_DIRECTORY,
    DatasetMetadata,
    drop_discontinuous_spans,
    local_datasets,
    read_monolingual_sample,
)
from meld.hf_dataset_conversion import convert_to_hf, dataset_filter, hf_config_name
from meld.manifest import DatasetManifest, load_label_map, load_manifest

logger = logging.getLogger("meld")

_TEMP_DIRECTORY = "downloads"

_MELD_OPEN = [
    "AgCNER",
    "AgriNER",
    "AnatEM",
    "BC2GM",
    "BC5CDR",
    "BioRED",
    "CANTEMIST",
    "CLEANANERCorp",
    "CrossNER",
    "E-NER",
    "FabNER",
    "Few-NERD",
    "FiNER-139",
    "FiNER-ORD",
    "FoNE",
    "German-LER",
    "Herodotos-Project-NER",
    "JNLPBA",
    "Japanese-Wikipedia",
    "MasakhaNER-X",
    "MultiCoNER",
    "MultiNERd",
    "NCBI-Disease",
    "NYTK-NerKor",
    "Naamapadam",
    "RaTE-NER",
    "SciREX",
    "SOFC-Exp",
    "SciER",
    "SoMeSci",
    "StackOverflowNER",
    "TASTEset",
    "Thai-NER",
    "Turku-NER-corpus",
    "Tweebank-NER",
    "UniversalNER",
    "WIESP2022",
    "WLP",
    "WNUT2017",
    "Weibo-NER",
    "WikiNEuRal",
    "idner-news-2k",
    "pioNER",
]

_MELD_NON_PROPRIETARY_EVAL = [
    *_MELD_OPEN,
    "Arabic-Cross-Dialectal-NER",
    "BC4CHEMD",
    "DanfeNER",
    "EBM-NLP",
    "EverestNER",
    "FindVehicle",
    "HarveyNER",
    "LegalNERo",
    "MIT-Movie",
    "MIT-Restaurant",
    "PhoNER-COVID19",
    "SCIERC",
    "TurkuONE",
    "TweetNER7",
    "WikiANN",
]

_MELD_NON_PROPRIETARY = [
    *_MELD_NON_PROPRIETARY_EVAL,
    "Polyglot-NER",
]

_MELD_PROFILES = {
    "meld:open": _MELD_OPEN,
    "meld:non-proprietary-eval": _MELD_NON_PROPRIETARY_EVAL,
    "meld:non-proprietary": _MELD_NON_PROPRIETARY,
    "meld:full": [*_MELD_NON_PROPRIETARY, "CoNLL-2003"],
}


def _download_preprocessed(data_directory: Path, meld_open_repo: str, datasets: set[str] | None) -> None:
    """
    Retrieve already processed MELD datasets from a HuggingFace Hub repository and convert it to the internal MELD format

    Args:
        data_directory: Root directory of the benchmark data.
        meld_open_repo: Identifier of the Hugging Face dataset repository to download from
            (e.g. `"kgnlp/meld-open"`). The repository must contain a
            dataset card with `meld_metadata` describing each dataset.
        datasets: Set of dataset names to extract from the repository.
            If `None`, all available datasets in the `meld_open_repo` are downloaded
    """
    logger.info(f"Downloading preprocessed datasets from {meld_open_repo}")

    dataset_card = DatasetCard.load(meld_open_repo, repo_type="dataset")
    configs: list = dataset_card.data["configs"]
    meld_metadata: dict = dataset_card.data["meld_metadata"]

    if datasets is None:
        data_to_load = meld_metadata
    else:
        data_to_load = datasets & meld_metadata.keys()

    files_to_download = []
    for config in configs:
        name = config["config_name"]
        if name.split("--", 1)[0] not in data_to_load:
            continue

        for split in config["data_files"]:
            files_to_download.extend(split["path"])

    download_path = data_directory / _TEMP_DIRECTORY / os.path.basename(meld_open_repo)
    processed_path = data_directory / PROCESSED_DIRECTORY
    processed_path.mkdir(exist_ok=True)

    snapshot_download(meld_open_repo, repo_type="dataset", local_dir=download_path, allow_patterns=files_to_download)

    metadata_adapter = TypeAdapter(DatasetMetadata)
    for dataset in data_to_load:
        metadata = metadata_adapter.validate_python(meld_metadata[dataset])
        dataset_path = processed_path / dataset
        dataset_path.mkdir(exist_ok=True)

        for subset in metadata._iter_subsets():
            subset_download_path = download_path / hf_config_name(dataset, subset.hierarchy)
            for split in subset.splits.values():
                output_path = dataset_path / split.path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pl.scan_parquet(subset_download_path / output_path.name).with_columns(
                    pl.col("sequence_id").str.replace_all("-", "").str.decode("hex")
                ).sink_parquet(output_path)

        metadata.dump(dataset_path)


def _resolve_profiles(datasets: Sequence[str] | None, manifest: DatasetManifest | None = None) -> set[str]:
    """
    Expand user‑provided dataset identifiers and MELD profiles into a concrete
    set of dataset names that exist in the benchmark manifest.

    Args:
        datasets: ``None`` to request *all* datasets supported by MELD, or a sequence
            containing individual dataset names, or profile shortcuts such as `"meld:open"`.
        manifest: Optional pre‑loaded `DatasetManifest`. If omitted, the built-in manifest is loaded from package data.

    Returns:
        A `set` of unique dataset names from the given datasets and profiles

    Raises:
        ValueError: If any name (after profile expansion) does not appear in `manifest`

    Example:
        >>> sorted(_resolve_profiles(["meld:open", "CoNLL-2003"]))
        ['AgCNER', 'AgriNER', 'AnatEM', 'BC2GM', 'BC5CDR', 'BioRED', 'CANTEMIST', 'CLEANANERCorp', 'CoNLL-2003', 'CrossNER', 'E-NER', 'FabNER', 'Few-NERD', 'FiNER-139', 'FiNER-ORD', 'FoNE', 'German-LER', 'Herodotos-Project-NER', 'JNLPBA', 'Japanese-Wikipedia', 'MasakhaNER-X', 'MultiCoNER', 'MultiNERd', 'NCBI-Disease', 'NYTK-NerKor', 'Naamapadam', 'RaTE-NER', 'SOFC-Exp', 'SciER', 'SciREX', 'SoMeSci', 'StackOverflowNER', 'TASTEset', 'Thai-NER', 'Turku-NER-corpus', 'Tweebank-NER', 'UniversalNER', 'WIESP2022', 'WLP', 'WNUT2017', 'Weibo-NER', 'WikiNEuRal', 'idner-news-2k', 'pioNER']
    """
    if manifest is None:
        manifest = load_manifest()
    available_datasets = manifest.keys()

    if datasets is None:
        return set(available_datasets)

    # Merge manually configured datasets and requested profiles
    requested_datasets = {
        dataset
        for dataset_or_profile in datasets
        for dataset in _MELD_PROFILES.get(dataset_or_profile, [dataset_or_profile])
    }
    unknown_datasets = requested_datasets - available_datasets
    if unknown_datasets:
        raise ValueError(f"Some of the requested datasets are not available in MELD: {unknown_datasets!r}")

    return requested_datasets


def download(
    data_directory: Path,
    datasets: Sequence[str] | None = None,
    force_reprocess: bool = False,
    meld_open_repo: str | None = "kgnlp/meld-open",
    sentence_span_path: Path | None = None,
) -> None:
    """
    Downloads NER datasets and processes them into the standardized benchmark format in the specified directory.

    Args:
        data_directory: Directory where the datasets will be stored.
        datasets: List of dataset names and/or profiles to download. Dataset profiles and names may be mixed. For example, ["meld:open", "CoNLL-2003"] will download all datasets in the "meld:open" list and "CoNLL-2003". If `None`, all available datasets will be downloaded.

        force_reprocess: Whether to reprocess the datasets even if they are already processed on disk.
        meld_open_repo: Repository ID on Huggingface Hub or path to preprocessed datasets in MELD format which will be loaded directly, bypassing processing from source for these datasets.
            If set to `None`, all datasets will be downloaded and processed from their original source.
        sentence_span_path: Reproduces the sentence tokenization bundled
            with the package and stores spans for each sentence in the
            given directory. Intended for full reproducibility and
            addition of new datasets.
    """
    manifest = load_manifest()
    requested_datasets = _resolve_profiles(datasets, manifest)

    # Resolve and create the cache directory if it doesn't already exist
    data_directory = data_directory.resolve()
    data_directory.mkdir(exist_ok=True)
    shared_intermediate_directory = data_directory / ".shared_download"

    preprocessed_datasets = set()
    if meld_open_repo is not None:
        _download_preprocessed(data_directory, meld_open_repo, requested_datasets)

    # Construct all data pipes in advance
    logger.info("Constructing data pipes")
    data_pipes = [
        (name, dataset_pipe(dataset), dataset.use_shared_cache)
        for name, dataset in manifest.items()
        if name in requested_datasets and name not in preprocessed_datasets
    ]
    logger.info(f'Downloading benchmark datasets to "{data_directory}"')

    if sentence_span_path is not None:
        sentence_span_path.mkdir(exist_ok=True)

    with logging_redirect_tqdm():
        for name, data_pipe, shared_cache_directory in data_pipes:
            processed_path = data_directory / PROCESSED_DIRECTORY / name
            if not force_reprocess and (processed_path / METADATA_FILENAME).exists():
                logger.info(f"{name} was already processed, skipping")
                continue

            logger.info(f"Processing {name}")
            (data_directory / PROCESSED_DIRECTORY).mkdir(exist_ok=True)

            if shared_cache_directory is None:
                intermediate_directory = data_directory / _TEMP_DIRECTORY / name
            else:
                shared_intermediate_directory.mkdir(exist_ok=True)
                intermediate_directory = shared_intermediate_directory / shared_cache_directory

            data_pipe.run(name, intermediate_directory, processed_path, sentence_span_path)


def sample_data(
    data_directory: Path,
    language: str,
    subset_size: int,
    output: Path | IO[bytes] | None = None,
    split: str = "train",
    tagset_config: dict[str, str] | None = None,
    merge_documents: bool = False,
    keep_documents_without_entities: bool = True,
    keep_discontinuous_spans: bool = False,
    target_num_tokens: int | None = None,
    aggregation_tokenizer: str = "google/gemma-3-27b-it",
) -> None:
    """
    Samples and processes data from a specified directory.

    Args:
        data_directory: The path to the directory containing the
            benchmark data.
        language: The ISO 639-3 code of the target language to sample.
        subset_size: The number of samples to extract per dataset.
        output: The destination for the output, either a file path or a
            writable IO object. Defaults to standard output.
        split: The dataset split to process, e.g., "train", "validation"
        tagset_config: Indicates which tagset to use for datasets with
            multiple tag sets for each sample. E.g. `{"Few-NERD":
            "fine"}` selects fine-grained tags from the `Few-NERD`
            dataset. This parameter is required if a dataset with
            multiple tagsets is encountered during sampling with the
            given configuration.
        merge_documents: Whether to merge documents consisting of
            multiple sentences or paragraphs into a single sample.
        keep_documents_without_entities: Whether to keep documents
            without entities.
        keep_discontinuous_spans: Whether to keep discontinuous spans.
            By default, only continuous spans are kept and flattened
            into simplified span annotations.
        target_num_tokens: If `merge_documents` is true, attempts to
            merge sentences or passages into documents only if the given
            number of tokens is not exceeded.
        aggregation_tokenizer: Tokenizer used for counting tokens if
            `target_num_tokens` is set.
    """

    sample = read_monolingual_sample(
        data_directory,
        language,
        subset_size,
        split,
        tagset_config,
        merge_documents,
        keep_documents_without_entities,
        target_num_tokens=target_num_tokens,
        aggregation_tokenizer=aggregation_tokenizer,
    )

    if not keep_discontinuous_spans:
        sample = drop_discontinuous_spans(sample)

    sample.sink_parquet(sys.stdout.buffer if output is None else output)


def available_datasets() -> list[str]:
    """
    Lists all available datasets.

    Returns:
        A list containing the names of all datasets included in the
        package.
    """

    return sorted(load_manifest().keys())


def _raise_if_none(bibtex: str | None) -> str:
    if bibtex is None:
        raise ValueError("Unexpected undefined bibtex raw found")

    return bibtex


_MELD_CITEKEY = "glocker2026meld"
_DEFAULT_CITEKEY = "PhoNER_COVID19"


def bibliography_entries(datasets: list[str] | None = None) -> list[str]:
    """
    Collects a list of bibliography entries as bibtex strings for the given datasets or MELD.

    Args:
        datasets: A list of datasets to collect bibliography entries for or `None` to return all dataset bibliography entries.

    Returns:
        A list of bibtex strings. If `None`, bibliography entries are returned for all datasets, if the passed list is empty,
        only the MELD entry is returned, otherwise all entries for the given list of datasets are returned in order
    """

    bibliography = bibtexparser.parse_string(  # pyright: ignore
        resources.files("meld.package_data").joinpath("dataset_references.bib").read_text()
    )
    manifest = load_manifest()

    if datasets is None:
        meld_citekey = _raise_if_none(bibliography.entries_dict[_MELD_CITEKEY].raw)
        bibliography.remove(bibliography.entries_dict[_MELD_CITEKEY])
        return [meld_citekey] + [_raise_if_none(entry.raw) for entry in bibliography.entries]

    if not datasets:
        return [
            _raise_if_none(bibliography.entries_dict[_MELD_CITEKEY].raw),
            f"% When using the PhoNER COVID19 dataset, also cite:\n{_raise_if_none(bibliography.entries_dict[_DEFAULT_CITEKEY].raw)}",
        ]

    bibtex = []
    for dataset in datasets:
        bibtex.extend(
            _raise_if_none(bibliography.entries_dict[cite_key].raw) for cite_key in manifest[dataset].citekeys
        )

    return bibtex


def _csv_list(value: str) -> list[str]:
    """
    Converts a comma-separated string into a list.

    Args:
        value: A comma-separated string.

    Raises:
        ArgumentTypeError: If the input is not a valid CSV string.

    Returns:
        A list of strings from the input CSV string.
    """

    try:
        return next(csv.reader([value]))
    except csv.Error as error:
        raise ArgumentTypeError(str(error))


def merge_data(
    data_directory: Path,
    output: Path | IO[bytes] | None = None,
    label_config: dict[str, str] | None = None,
    merge_documents: bool = False,
) -> None:
    """
    Merges data from multiple datasets into a single parquet output.

    Args:
        data_directory: Directory containing processed benchmark
            datasets.
        output: Output path for merged data, or stdout if None.
        label_config: Configuration mapping dataset names to their
            tagsets for multi-tagset datasets.
        merge_documents: Whether to merge multiple sentences/paragraphs
            into single documents.
    """

    data_splits: list[pl.LazyFrame] = []

    for dataset in formats.local_datasets(data_directory):
        dataset_name = dataset.metadata.name
        for subset in dataset:
            for split in subset.splits:
                tagsets = subset.metadata.tagsets
                columns_to_rename = {}

                if label_config is None:
                    columns_to_remove = tagsets
                elif len(tagsets) > 1:
                    try:
                        selected_tagset = label_config[dataset_name]
                    except KeyError:
                        raise KeyError(
                            f"No tagset specified for {dataset_name}, which contains multiple tagsets ({tagsets})"
                        )
                    columns_to_remove = [tagset for tagset in tagsets if tagset != selected_tagset]
                    columns_to_rename[selected_tagset] = DEFAULT_TAGSET
                    if subset.metadata.pre_tokenized:
                        columns_to_rename[f"{selected_tagset}_iob"] = f"{DEFAULT_TAGSET}_iob"
                else:
                    columns_to_remove = []

                reorder_tokens = False
                if subset.metadata.pre_tokenized:
                    columns_to_add = {}
                    if columns_to_remove:
                        columns_to_remove = columns_to_remove.copy()
                        # Store in a list first to avoid recursion
                        columns_to_remove.extend([f"{tagset}_iob" for tagset in columns_to_remove])
                else:
                    columns_to_add = {"tokens": pl.lit(None, pl.List(pl.String()))}
                    if label_config is not None:
                        reorder_tokens = True
                        columns_to_add[f"{DEFAULT_TAGSET}_iob"] = pl.lit(None, pl.List(pl.String()))

                data = (
                    subset.scan_split(split)
                    .with_columns(
                        **columns_to_add,
                        dataset=pl.lit(dataset_name),
                        subset=pl.lit("/".join(subset.hierarchy)),
                        split=pl.lit(split),
                        language=pl.lit(subset.metadata.language),
                    )
                    .drop(*columns_to_remove)
                    .rename(columns_to_rename)
                )

                # Note: Column order matters for strict concatenation in polars. Therefore, the "tokens" and "ner" tagset column are swapped if empty tokens columns are inserted
                if reorder_tokens:
                    columns = data.collect_schema().names()
                    tokens_index = columns.index("tokens")
                    columns[tokens_index - 1 : tokens_index + 1] = columns[tokens_index : tokens_index - 2 : -1]
                    data = data.select(*columns)

                if merge_documents:
                    data = formats._merge_documents(data).with_columns(
                        dataset=pl.lit(dataset_name),
                        subset=pl.lit("/".join(subset.hierarchy)),
                        split=pl.lit(split),
                        language=pl.lit(subset.metadata.language),
                    )

                # Collect each split directly to avoid memory spikes
                data_splits.append(data)

    # Limited chunk size to reduce memory usage
    with pl.Config(streaming_chunk_size=512):
        pl.concat(data_splits).sink_parquet(sys.stdout.buffer if output is None else output)


_WORD_COUNT_BATCH_SIZE = 1024


def compute_word_counts(data_directory: Path, output: Path, append: bool = False, workers: int = 12) -> None:
    """
    Computes word counts using a word tokenizer for each dataset split and writes statistics to a parquet file.

    Args:
        data_directory: Directory containing processed benchmark
            datasets.
        output: Path to output parquet file for word count statistics.
        append: Whether to append to an existing output file instead of
            overwriting.
        workers: Number of workers to use for parallel word tokenization.
            Note that a high worker count will increase memory consumption substantially for some tokenizers
    """

    if append:
        previous_data = pl.read_parquet(output)
        processed_datasets = set(previous_data.select(pl.col("dataset_name").unique()).to_series())
        statistics = iter(
            (
                stats
                for dataset in local_datasets(data_directory)
                if dataset.metadata.name not in processed_datasets
                for stats in data_stats.word_tokenize_dataset(dataset, workers)
            ),
        )

    else:
        previous_data = None
        statistics = iter(
            stats
            for dataset in local_datasets(data_directory)
            for stats in data_stats.word_tokenize_dataset(dataset, workers)
        )

    first_row = next(statistics)
    first_sample = dataclasses.asdict(first_row)
    first_batch = pa.RecordBatch.from_pylist([first_sample])
    schema = first_batch.schema
    subset_field = schema.get_field_index("subset_hierarchy")
    schema = schema.set(subset_field, schema.field(subset_field).with_type(pa.list_(pa.string())))

    first_batch = first_batch.cast(schema)

    batch = [first_sample]

    with ParquetWriter(output, schema=schema, compression="zstd") as writer, logging_redirect_tqdm():
        if previous_data is not None:
            writer.write_table(previous_data.to_arrow().cast(schema))

        for split_statistics in tqdm(statistics):
            batch.append(dataclasses.asdict(split_statistics))
            if len(batch) >= _WORD_COUNT_BATCH_SIZE:
                writer.write_batch(pa.RecordBatch.from_pylist(batch, schema=schema))
                batch = []

        # Write final batch
        if batch:
            writer.write_batch(pa.RecordBatch.from_pylist(batch, schema=schema))


def _label_config_type(config: str) -> dict[str, str] | None:
    """
    Parses a JSON string into a dictionary mapping dataset names to tagsets and treats "omit" as `None`.

    Args:
        config: JSON string in format {"dataset": "tagset"} or literal
            "omit".

    Returns:
        Dictionary mapping dataset names to tagset names, or None if
        "omit" is provided.

    Raises:
        ArgumentTypeError: If JSON parsing fails.
    """

    config_parser = TypeAdapter(dict[str, str])
    try:
        return config_parser.validate_json(config)
    except ValidationError as error:
        raise ArgumentTypeError(
            'Config has to be either "omit" or valid JSON of the form {"dataset": "tagset"}'
        ) from error


type JsonType = dict[str, JsonType] | list[JsonType] | str | int | float | bool | None


def _json_type(json_string: str) -> JsonType:
    """
    Parse a JSON string into a Python object and handle errors
    as `ArgumentTypeError` for use with `argparse`.

    Args:
        json_string: A string containing JSON data.

    Returns:
        The Python object parsed from the string.

    Raises:
        ArgumentTypeError: If the input is not valid JSON.
    """
    try:
        return json.loads(json_string)
    except JSONDecodeError as error:
        raise ArgumentTypeError(f"Invalid JSON: {error}")


def _argument_parser() -> ArgumentParser:
    """
    Creates and returns the CLI argument parser for the MELD data management tool.

    Returns:
        An ArgumentParser configured with subcommands for download,
        list, sample, merge, and count-words operations.
    """

    log_levels = logging.getLevelNamesMapping()

    parser = ArgumentParser(description="Manage datasets of MELD")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    download_parser = subparsers.add_parser("download", description="Download MELD", help="Download MELD")
    download_parser.add_argument(
        "data_directory",
        nargs="?",
        type=Path,
        default=Path("./meld_data"),
        help="Directory where benchmark data will be downloaded",
    )
    download_parser.add_argument(
        "-d",
        "--datasets",
        type=_csv_list,
        default=("meld:open",),
        help=(
            "Comma-separated list of datasets or pre-defined subsets to download If not provided, the meld:open subset containing all redistributable datasets will be downloaded. "
            f"The following pre-defined subsets are available: {', '.join(_MELD_PROFILES)}"
        ),
    )
    download_parser.add_argument(
        "-v",
        "--log-level",
        metavar="LEVEL",
        choices=log_levels.keys(),
        type=str.upper,
        default="WARN",
        help="Set the log level for the download command",
    )
    download_parser.add_argument(
        "--sentence-span-path",
        type=Path,
        help=(
            "Reproduces the sentence tokenization bundled with the package "
            "and stores spans for each sentence in the given directory. "
            "Intended for full reproducibility and addition of new datasets."
        ),
    )
    download_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-process all datasets. If not given, previously processed datasets are kept",
    )
    group = download_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--meld-repo",
        default="kgnlp/meld-open",
        help="Repository ID on Huggingface Hub or path to preprocessed datasets in MELD format which will be loaded directly, bypassing processing from source for these datasets.",
    )
    group.add_argument(
        "--reproduce",
        action="store_const",
        help="Reproduces MELD by processing all datasets from source, including those for which a preprocessed version is available on the Huggingface Hub",
        dest="meld_repo",
    )

    subparsers.add_parser(
        "list",
        description="List all datasets available in MELD",
        help="Display the list of available datasets",
    )

    sample_parser = subparsers.add_parser(
        "sample",
        description="Generate a monolingual sample of each dataset in the benchmark",
        help="Geenrate a monolingual sample of each dataset",
    )
    sample_parser.add_argument(
        "data_directory",
        type=Path,
        help="Directory from which benchmark data will be sampled",
    )
    sample_parser.add_argument("language", help="ISO639-3 code of the language to sample")
    sample_parser.add_argument("subset_size", type=int, help="Number of samples from each dataset")
    sample_parser.add_argument(
        "-l",
        "--label-config",
        type=_label_config_type,
        help='Selection of tag sets in JSON format for datasets with multiple tagsets for each sample. E.g. {"Few-NERD": "fine"} selects fine-grained tags from the Few-NERD dataset',
    )
    sample_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=sys.stdout.buffer,
        help="Output path for the generated sample in parquet format",
    )
    sample_parser.add_argument(
        "-m",
        "--merge-documents",
        action="store_true",
        help="Merges documents from sentence-tokenized datasets into single texts",
    )
    sample_parser.add_argument(
        "-k",
        "--keep-documents-without-entities",
        action="store_true",
        help="Keeps sentences, passages or documents that do not contain any entity annotations. If -m/--merge-documents was set, sentences without annotations will be kept if they are a part of a document that contains annotations in other sentences.",
    )
    sample_parser.add_argument(
        "-t",
        "--merging-max-tokens",
        type=int,
        help="Sets a maximum number of tokens per document during merging. If the maximum is exceeded, documents will be merged in shorter chunks",
    )
    sample_parser.add_argument(
        "--tokenizer",
        default="google/gemma-3-27b-it",
        help="The tokenizer to use for counting tokens if `-t/--merging-max-tokens` is set. Either a HuggingFace Hub ID or a path",
    )
    sample_parser.add_argument(
        "-s",
        "--split",
        default="train",
        help='The name of the split from which to sample from each dataset, defaults to "train"',
    )
    sample_parser.add_argument(
        "--keep-discontinuous-spans", action="store_true", help="Keeps annotations of discontinuous spans"
    )

    merge_parser = subparsers.add_parser("merge", description="Merge benchmark data into a single file")
    merge_parser.add_argument(
        "data_directory",
        type=Path,
        help="Directory from which benchmark data will be sampled",
    )
    merge_parser.add_argument(
        "-l",
        "--label-config",
        type=_label_config_type,
        help='Enables tags in the merged output when specifying tag sets in JSON format for datasets with multiple tag sets for each sample. If this argument is not included, the merged dataset will not contain labels. E.g. {"Few-NERD": "fine"} selects fine-grained tags from the Few-NERD dataset',
    )
    merge_parser.add_argument(
        "-m",
        "--merge-documents",
        action="store_true",
        help="Merges documents from sentence-tokenized datasets into single texts. Note that document merging will currently remove tokenized text and IOB annotations where available.",
    )
    merge_parser.add_argument(
        "-o", "--output", type=Path, default=sys.stdout.buffer, help="Output path for the merged data in parquet format"
    )

    count_words_parser = subparsers.add_parser("count-words", description="Compute token counts for each dataset split")
    count_words_parser.add_argument(
        "data_directory",
        type=Path,
        help="Directory from which benchmark data will be tokenized",
    )
    count_words_parser.add_argument(
        "output",
        type=Path,
        help="Output path for token count statistics in parquet format",
    )
    count_words_parser.add_argument(
        "-v",
        "--log-level",
        metavar="LEVEL",
        choices=log_levels.keys(),
        type=str.upper,
        default="WARN",
        help="Set the log level for tokenization",
    )
    count_words_parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Appends to the give output file and only processes datasets that were not already in the processed file",
    )
    count_words_parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=12,
        help=(
            "Number of workers to use for parallel word tokenization. "
            "Note that a high worker count will increase memory consumption substantially for some tokenizers"
        ),
    )

    cite_parser = subparsers.add_parser(
        "cite", description="Prints bibtex bibliography entries for meld or the chosen datasets"
    )
    group = cite_parser.add_mutually_exclusive_group()
    group.add_argument(
        "datasets", type=_csv_list, nargs="?", help="Comma-separated list of datasets to print bibliography entries for"
    )
    group.add_argument("--all", action="store_true", help="Print bibtex entries for all datasets in MELD")

    hf_conversion_parser = subparsers.add_parser("hf", description="Convert a MELD datasets to HuggingFace Hub format")
    hf_conversion_parser.add_argument("meld_data_path", type=str, help="Path to the MELD dataset directory")
    hf_conversion_parser.add_argument(
        "output_data_path", type=str, help="Output directory for the converted HF dataset"
    )
    hf_conversion_parser.add_argument(
        "-n",
        "--normalize-labels",
        metavar="JSON_FILE",
        nargs="?",
        const="",
        help="Will normalize entity type labels either with the built-in entity type mapping a custom tag mapping JSON file",
    )
    hf_conversion_parser.add_argument(
        "-d",
        "--datasets",
        type=_csv_list,
        default=("meld:open",),
        help=(
            "Comma-separated list of datasets or pre-defined subsets to include in the dataset. If not provided, only redistributable datasets will be included. "
            f"The following pre-defined subsets are available: {', '.join(_MELD_PROFILES)}"
        ),
    )
    hf_conversion_parser.add_argument(
        "-t", "--dataset-card-template", type=Path, help="Path to a custom dataset card template file"
    )
    hf_conversion_parser.add_argument("-p", "--pretty-name", help="Human-readable name for the dataset card")
    hf_conversion_parser.add_argument(
        "-e",
        "--extra-files",
        type=_json_type,
        help="JSON object specifying lists of of additional files per dataset to include in their output directories, such as licenses",
    )

    return parser


def main(args: Sequence[str] | None = None) -> None:
    """
    Main entry point for the MELD data management CLI.

    Args:
        args: Command line arguments. If None, arguments are parsed from
            `sys.argv`.
    """

    if args is None:
        args = sys.argv[1:]

    arguments = _argument_parser().parse_args(args)
    log_levels = logging.getLevelNamesMapping()

    match arguments.mode:
        case "list":
            for dataset in available_datasets():
                print(dataset)
        case "cite":
            for entry in bibliography_entries(None if arguments.all else (arguments.datasets or [])):
                print(entry)
        case "download":
            logger.setLevel(log_levels[arguments.log_level])
            download(
                arguments.data_directory,
                arguments.datasets,
                arguments.force,
                arguments.meld_repo,
                arguments.sentence_span_path,
            )
        case "sample":
            sample_data(
                arguments.data_directory,
                arguments.language,
                arguments.subset_size,
                arguments.output,
                arguments.split,
                arguments.label_config,
                arguments.merge_documents,
                arguments.keep_documents_without_entities,
                arguments.keep_discontinuous_spans,
                arguments.merging_max_tokens,
                arguments.tokenizer,
            )
        case "merge":
            merge_data(arguments.data_directory, arguments.output, arguments.label_config, arguments.merge_documents)
        case "count-words":
            logger.setLevel(log_levels[arguments.log_level])
            compute_word_counts(arguments.data_directory, arguments.output, arguments.append, arguments.workers)
        case "hf":
            match arguments.normalize_labels:
                case "":
                    label_mapping = load_label_map()
                case None:
                    label_mapping = None
                case _:
                    label_mapping = load_label_map(arguments.normalize_labels)

            included_datasets = _resolve_profiles(arguments.datasets)

            with open(arguments.dataset_card_template) as file:
                dataset_card_template = file.read()

            convert_to_hf(
                arguments.meld_data_path,
                arguments.output_data_path,
                None if arguments.datasets is None else dataset_filter(included_datasets),
                label_mapping,
                dataset_card_template,
                arguments.pretty_name,
                arguments.extra_files,
            )


if __name__ == "__main__":
    main()
