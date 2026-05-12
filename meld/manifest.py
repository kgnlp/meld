"""Dataset manifest and configuration classes."""

import json
from dataclasses import field
from importlib import resources
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BeforeValidator, ConfigDict, Field, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass

from meld.document import BIOField

_strict_config = ConfigDict(extra="forbid")


@dataclass(config=_strict_config)
class URLWithChecksum:
    """
    A URL with its SHA256 checksum for verification.

    Attributes:
        url: The URL to download from.
        sha256: The expected SHA256 checksum of the downloaded file.
    """

    url: str
    sha256: str


@dataclass(config=_strict_config)
class DownloadStep:
    """
    A data pipe step that downloads files from URLs with checksum verification.

    Attributes:
        step: The step type identifier ("download").
        urls: List of URLs with their expected checksums.
    """

    step: Literal["download"]
    urls: list[URLWithChecksum]


@dataclass(config=_strict_config)
class URLWithTarget:
    """
    A URL with target filename and SHA256 checksum for Google Docs downloads.

    Attributes:
        url: The Google Docs URL to download from.
        target_filename: The filename to save the downloaded file as.
        sha256: The expected SHA256 checksum of the downloaded file.
    """

    url: str
    target_filename: str
    sha256: str


@dataclass(config=_strict_config)
class GoogleDocsStep:
    """
    A data pipe step that downloads files from Google Docs.

    Attributes:
        step: The step type identifier ("google_docs").
        urls: List of Google Docs URLs with target filenames and
            checksums.
    """

    step: Literal["google_docs"]
    urls: list[URLWithTarget]


@dataclass(config=_strict_config)
class GitStep:
    """
    A data pipe step that clones a Git repository and extracts the given files.

    Attributes:
        step: The step type identifier ("git").
        repo: The Git repository URL to clone.
        revision: The commit hash to checkout.
        files: List of file paths to extract from the repository.
            Relative to `directory`.
        directory: Optional base directory to which the paths in `files`
            are relative.
        keep_repo: Whether to keep the cloned repository on disk after
            extraction.
    """

    step: Literal["git"]
    repo: str
    revision: str
    files: list[str]
    directory: str = ""
    keep_repo: bool = False


@dataclass(config=_strict_config)
class ExtractStep:
    """
    A step that extracts files from a compressed archive.

    Attributes:
        step: The step type identifier ("extract").
        from_file: The compressed file to extract from.
        files: List of file paths to extract.
        use_globs: Whether to treat file paths as glob patterns.
    """

    step: Literal["extract"]
    from_file: str
    files: list[str]
    use_globs: bool = False


type SplitFiles = list[str] | list[dict[str, list[str]]]


@dataclass(config=_strict_config)
class SplitStep:
    """
    A data pipe step that splits data into train/validation/test sets.

    Attributes:
        step: The step type identifier ("splits").
        language: The language of the data. This should be an ISO 639-3
            language code if possible.
        directory: Optional subdirectory containing the data files.
        train: Training split file specifications.
        validation: Validation split file specifications.
        test: Test split file specifications.
        subset: Optional subset name within the data.
        metadata: Additional metadata for this split.
    """

    step: Literal["splits"]
    language: str
    directory: str = ""
    train: SplitFiles = field(default_factory=list)
    validation: SplitFiles = field(default_factory=list)
    test: SplitFiles = field(default_factory=list)
    subset: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @model_validator(mode="after")
    def validate_splits(self) -> Self:
        """
        Validates that at least one split is defined.

        Returns:
            The `SplitStep` itself.

        Raises:
            ValueError: If no splits are defined.
        """

        if not any((self.train, self.validation, self.test)):
            raise ValueError("At least one split must be defined with at least one file")
        return self


@dataclass(config=_strict_config)
class ReadSplitStep:
    """
    A data pipe step that splits data based on external metadata files.

    Attributes:
        step: The step type identifier ("read_splits").
        language: The language of the data. This should be an ISO 639-3
            language code if possible.
        split_files: Path or mapping of split names to file paths.
        splits_reader: The split metadata reader implementation to use.
        directories: Optional directory or mapping of directories for
            files.
        split_name_map: Optional mapping to rename splits.
        subset: Optional subset name within the data.
        metadata: Additional metadata for this step.
    """

    step: Literal["read_splits"]
    language: str
    split_files: str | dict[str, str]
    splits_reader: Literal[
        "legalnero_standoff_split_columns", "agriner_standoff_split_json", "somesci_standoff_split_json"
    ]
    directories: str | dict[str, str] = ""
    split_name_map: dict[str, str] | None = None
    subset: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


type CoNLLDialectNames = Literal[
    "conll",
    "conll_bio_first",
    "ner_suite",
    "flat",
    "conll2003",
    "conll2003_ignore_docstart",
    "conll2003_two_column",
    "conll2003_pioner",
    "conll_jnlpba",
    "conll_weibo_ner",
    "conll_stackoverflow_ner",
    "conllu_uner",
    "conllu_uner_newpar",
    "conll_herodotos",
    "conll_with_pos",
    "conllu_plus",
]


@dataclass(config=_strict_config)
class CoNLLArguments:
    """
    Configuration arguments for CoNLL format data processing.

    Attributes:
        shards_are_documents: Whether each shard file represents a
            single, complete document.
        dialect: The CoNLL dialect variant to use.
        delimiter: Field delimiter in the CoNLL file.
        label_map: Optional mapping of tagsets to label index mappings
            for CoNLL-style data that uses tag indices.
        bioes_to_bio: Whether to convert BIOES tags to BIO format.
        enforce_blank_lines: Whether to enforce that blank lines between
            sentences do not contain whitespace.
        preprocessor: Optional preprocessor for specific datasets that
            is run on the raw data before parsing.
    """

    shards_are_documents: bool = False
    # ner_suite for AnatEM, flat for flat columns, conll_bio_first for MIT-* datasets, conll2003_ignore_docstart for Tweebank-NER, conll2003_two_column for HarveyNER
    dialect: CoNLLDialectNames = "conll"
    delimiter: str = "\t"
    label_map: dict[str, dict[BIOField, int]] | None = None
    bioes_to_bio: bool = False
    enforce_blank_lines: bool = True
    preprocessor: Literal["e-ner", "stackoverflow_ner", "pioner", "nytk_nerkor"] | None = None


@dataclass(config=_strict_config)
class OffsetCSVArguments:
    """
    Configuration arguments for offset-based CSV data.

    Attributes:
        text_column: Name of the column containing the text.
        offsets_column: Name of the column containing character offsets
            for entity span annotations.
    """

    text_column: str
    offsets_column: str


@dataclass(config=_strict_config)
class ByteOffsetJSONLArguments:
    """
    Configuration arguments for byte offset-based JSONL data.

    Attributes:
        text_key: Key for the text field in JSON objects.
        offsets_key: Key for the byte offsets field in JSON objects.
        annotated_span_target_key: Optional key containing expected
            string representations for each span for validation.
    """

    text_key: str
    offsets_key: str
    annotated_span_target_key: str | None = None


@dataclass(config=_strict_config)
class StandoffArguments:
    """
    Configuration arguments for standoff annotated data.

    Attributes:
        offsets_without_newlines: Whether to exclude newlines from
            offset calculations.
    """

    offsets_without_newlines: bool = False


@dataclass(config=_strict_config)
class SofcStandoffArguments:
    """
    Configuration arguments for the SOFC dataset's standoff-style format.

    Attributes:
        label_source: Source of labels to use as entity annotations for
            a given subset (frames or entities).
    """

    label_source: Literal["frames", "entities"]


@dataclass(config=_strict_config)
class EBMNLPStandoffArguments:
    """
    Configuration arguments for the EBM-NLP dataset's standoff format.

    Attributes:
        label_map: Mapping of label names to integer indices for each
            tagset.
        broad_label_map: Mapping of broad label names to integer indices
            for each tagset.
    """

    label_map: dict[str, dict[str, int]]
    broad_label_map: dict[str, dict[str, int]]


@dataclass(config=_strict_config)
class PlainSpanArguments:
    """
    Configuration arguments for plain span data.

    Attributes:
        span_format: Format of span annotations (JSON or Python
            dictionary style).
    """

    span_format: Literal["json", "python"] = "json"


type DetokenizerType = Literal["concatenate", "whitespace", "wikiann"]


@dataclass(config=_strict_config)
class CoNLLConfiguration:
    """
    Configuration for the CoNLL-style data reader.

    Attributes:
        type: The configuration type identifier ("conll").
        arguments: CoNLL-specific arguments.
        detokenizer_type: Strategy for detokenizing tokens.
    """

    type: Literal["conll"]
    arguments: CoNLLArguments = field(default_factory=CoNLLArguments)
    detokenizer_type: DetokenizerType = "whitespace"


@dataclass(config=_strict_config)
class OffsetCSVConfiguration:
    """
    Configuration for the offset-based CSV data reader.

    Attributes:
        type: The configuration type identifier ("offset_csv").
        arguments: OffsetCSV-specific arguments.
    """

    type: Literal["offset_csv"]
    arguments: OffsetCSVArguments


@dataclass(config=_strict_config)
class ByteOffsetJSONLConfiguration:
    """
    Configuration for the byte offset-based JSONL reader.

    Attributes:
        type: The configuration type identifier ("byte_offset_jsonl").
        arguments: ByteOffsetJSONL-specific arguments.
    """

    type: Literal["byte_offset_jsonl"]
    arguments: ByteOffsetJSONLArguments


@dataclass(config=_strict_config)
class StandoffConfiguration:
    """
    Configuration for the standoff annotation reader.

    Attributes:
        type: The configuration type identifier ("standoff").
        arguments: Standoff-specific arguments.
    """

    type: Literal["standoff"]
    arguments: StandoffArguments = field(default_factory=StandoffArguments)


@dataclass(config=_strict_config)
class SofcStandoffConfiguration:
    """
    Configuration for SOFC standoff reader.

    Attributes:
        type: The configuration type identifier ("sofc_standoff").
        arguments: SofcStandoff-specific arguments.
    """

    type: Literal["sofc_standoff"]
    arguments: SofcStandoffArguments


@dataclass(config=_strict_config)
class EBMNLPStandoffConfiguration:
    """
    Configuration for EBM-NLP standoff reader.

    Attributes:
        type: The configuration type identifier ("ebm_nlp_standoff").
        arguments: EBMNLPStandoff-specific arguments.
    """

    type: Literal["ebm_nlp_standoff"]
    arguments: EBMNLPStandoffArguments


@dataclass(config=_strict_config)
class PlainSpanConfiguration:
    """
    Configuration for plain span reader.

    Attributes:
        type: The configuration type identifier ("plain_spans").
        arguments: PlainSpan-specific arguments.
    """

    type: Literal["plain_spans"]
    arguments: PlainSpanArguments = field(default_factory=PlainSpanArguments)


@dataclass(config=_strict_config)
class ReaderConfigurationWithoutArguments:
    """
    Configuration for reader formats that require no additional arguments.

    Attributes:
        type: The configuration type identifier for the desired reader
            format.
    """

    type: Literal["bioc_xml", "pubtator", "scirex_jsonl", "scier_jsonl", "arabic_cross_dialectal_json", "dataset_spans"]


type ReaderConfiguration = (
    ReaderConfigurationWithoutArguments
    | CoNLLConfiguration
    | OffsetCSVConfiguration
    | ByteOffsetJSONLConfiguration
    | StandoffConfiguration
    | SofcStandoffConfiguration
    | EBMNLPStandoffConfiguration
    | PlainSpanConfiguration
)


@dataclass(config=_strict_config)
class ConvertStep:
    """
    A data pipe step that converts data from a source format to the normalized MELD format using a source format specific reader.

    Attributes:
        step: The step type identifier ("convert").
        reader: The reader configuration to use for parsing.
        filter_empty_documents: Whether to explicitly remove all documents with empty text received from the reader.
            If set to `False`, empty documents from the reader will raise an error.
    """

    step: Literal["convert"]
    reader: ReaderConfiguration
    filter_empty_documents: bool = False


@dataclass(config=_strict_config)
class NestedDataPipeStep:
    """
    A data pipe step that defines a nested data pipe.

    Attributes:
        step: The step type identifier ("data_pipe").
        data_pipe: List of other data pipe steps to execute, including
            potentially other nested data pipe steps.
    """

    step: Literal["data_pipe"]
    data_pipe: list["GenericDataPipeStep"]


type GenericDataPipeStep = (
    NestedDataPipeStep | DownloadStep | GoogleDocsStep | GitStep | ExtractStep | SplitStep | ReadSplitStep | ConvertStep
)


@dataclass(config=_strict_config)
class TagSet:
    """
    Configuration of a tagset for data conversion and normalization.

    Attributes:
        label_map: Optional mapping of labels to integer indices for
            handling formats where labels are represented by indices.
        column: Optional column name to use for this tagset.
    """

    label_map: dict[BIOField, int] | None = None
    column: str | None = None


@dataclass(config=_strict_config)
class DatasetConvertStep:
    """
    A step that converts Huggingface datasets to the normalized MELD format.

    Attributes:
        step: The step type identifier ("columns").
        tagsets: Mapping of tagset names to column names or TagSet
            configurations.
        sequence_type: Whether sequences should be treated as sentences
            or passages.
        detokenizer_type: Strategy for detokenizing text.
        bio_type: Type of BIO tags to parse. Options are "iob" (standard IOB format) or "iob_type_only" (IOB format without an "I-" or "B-" prefix). Defaults to "iob".
        filter_empty_documents: Whether to explicitly remove all documents with empty text received from the reader.
            If set to `False`, empty documents from the reader will raise an error.
    """

    step: Literal["columns"]
    tagsets: dict[str, str | TagSet] | None = None
    sequence_type: Literal["sentence", "passage"] = "sentence"
    detokenizer_type: DetokenizerType = "whitespace"
    bio_type: Literal["iob", "iob_type_only"] = "iob"
    filter_empty_documents: bool = False


type DataPipeStep = GenericDataPipeStep | DatasetConvertStep


@dataclass(config=_strict_config)
class SubsetDataFiles:
    """
    Configuration for dataset files organized by subset.

    Attributes:
        subsets: Mapping of subset names to split file specifications.
    """

    subsets: dict[str, dict[str, list[str] | str]]


@dataclass(config=_strict_config)
class HuggingfaceArguments:
    """
    Configuration arguments for loading data from Huggingface datasets.

    Note: Only one of language_column, language_from_subset_name, and base_language can be specified at a time.

    Attributes:
        repo: Huggingface dataset repository ID.
        revision: Repository version (ideally commit hash).
        text_column: Name of the text column in the dataset.
        tag_column: Name of the tag column in the dataset.
        train_name: Name of the training split.
        validation_name: Name of the validation split.
        test_name: Name of the test split.
        base_language: Base language code for the dataset. This should
            be an ISO 639-3 language code if possible.
        language_column: Optional name of the column containing language
            codes for splitting the dataset into language subsets. Will
            be converted to ISO 639-3 automatically, if possible.
        language_from_subset_name: Pattern to dynamically extract the
            language from subset names. Will be converted to ISO 639-3
            automatically, if possible.
        fast_subset_load: Whether to use optimized loading for datasets
            with many subsets (such as WikiANN).
        trust_remote_code: Whether to trust remote code execution.
        split_naming_pattern: Pattern for naming splits.
        data_pipe: Data processing data pipe steps to apply to the
            dataset.
        data_files: Manual data file specifications which will override
            the manual file discovery of the `datasets` library.
    """

    repo: str
    revision: str

    text_column: str
    tag_column: str

    train_name: str | None = None
    validation_name: str | None = None
    test_name: str | None = None

    base_language: str | None = None
    language_column: str | None = None

    language_from_subset_name: str | None = None
    fast_subset_load: bool = False
    trust_remote_code: bool = False
    split_naming_pattern: str | None = None
    data_pipe: list[DataPipeStep] = field(default_factory=lambda: [DatasetConvertStep("columns")])
    data_files: SubsetDataFiles | dict[str, list[str] | str] | None = None

    @model_validator(mode="after")
    def validate_base_language_and_splits(self) -> Self:
        """
        Validates that at least one split is configured and language configuration is consistent.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If no splits are named or if language
                configurations conflict.
        """

        if all(name is None for name in (self.train_name, self.validation_name, self.test_name)):
            raise ValueError("At least one split must be named")

        if (
            self.language_column is not None or self.language_from_subset_name is not None
        ) and self.base_language is not None:
            raise ValueError(
                "base_language must be null or unspecified if language_column or language_from_subset_name are configured"
            )

        return self


@dataclass(config=_strict_config)
class FileSubset:
    """
    Configuration for the processing data pipe and splits of a subset.

    Attributes:
        train: Training split file specifications.
        validation: Validation split file specifications.
        test: Test split file specifications.
        data_pipe: Optional data pipe for this specific subset. If not
            defined or set to `None`, the default_data_pipe in scope
            will be used.
        directory: Optional subdirectory containing the data files.
        language: Optional language code for this subset. This should be
            an ISO 639-3 language code if possible.
    """

    train: SplitFiles = field(default_factory=list)
    validation: SplitFiles = field(default_factory=list)
    test: SplitFiles = field(default_factory=list)
    data_pipe: Annotated[list[GenericDataPipeStep], Field(min_length=1)] | None = None
    directory: str = ""
    language: str | None = None


@dataclass(config=_strict_config)
class GitLoaderArguments:
    """
    Configuration arguments for downloading data from a Git repository.

    Attributes:
        repo: Git repository URL.
        revision: Repository version (preferably commit hash).
        subsets: Mapping of subset names to subset configurations.
        base_language: Base language code for the dataset. This should
            be an ISO 639-3 language code if possible.
        default_data_pipe: Default data pipe to use for processing. Each
            subset can override the default data pipe for subset-
            specific processing.
        keep_repo: Whether to keep the cloned repository after relevant
            files have been extracted.
    """

    repo: str
    revision: str
    subsets: dict[str, FileSubset]
    base_language: str | None = None
    default_data_pipe: list[GenericDataPipeStep] = field(default_factory=list)
    keep_repo: bool = False


@dataclass(config=_strict_config)
class GenericArguments:
    """
    Configuration arguments for generic data loading.

    Attributes:
        download_data_pipe: Data processing data pipe for downloading
            and processing data.
    """

    download_data_pipe: list[GenericDataPipeStep]


@dataclass(config=_strict_config)
class GenericLoader:
    """
    Loader configuration for generic data loading and processing from web sources or local files.

    Attributes:
        loader: The loader type identifier ("generic").
        arguments: Generic data processing arguments.
    """

    loader: Literal["generic"]
    arguments: GenericArguments


@dataclass(config=_strict_config)
class HuggingfaceLoader:
    """
    Loader configuration for HuggingFace datasets.

    Attributes:
        loader: The loader type identifier ("huggingface").
        arguments: HuggingFace data processing arguments.
    """

    loader: Literal["huggingface"]
    arguments: HuggingfaceArguments


@dataclass(config=_strict_config)
class GitLoader:
    """
    Loader configuration for Git repositories.

    Attributes:
        loader: The loader type identifier ("git").
        arguments: Git data processing arguments.
    """

    loader: Literal["git"]
    arguments: GitLoaderArguments


@dataclass(config=_strict_config)
class Format:
    """
    Metadata providing details about the source data format of the dataset.

    Attributes:
        text: Whether text is pre-tokenized or original documents are
            preserved.
        tags: Type of tag format (BIO, BIOES, spans, or discontinuous
            spans).
        text_properties: Additional metadata concerning the text content of the dataset.
        token_format: Optional tokenizer details for pre-tokenized
            datasets (e.g., WikiANN using hashes to represent whitespace
            in certain languages).
        tag_format: Optional tag format (e.g., whether tags are
            represented by indices).
        tag_alignment: How tags are aligned to text (offsets or byte
            offsets).
        token_alignment: How tokens are aligned to text .
        text_alignment: How text segments are aligned.
    """

    text: Literal["pre-tokenized", "original"]
    tags: Literal["bio", "bioes", "spans", "discontinuous_spans"]
    text_properties: list = field(default_factory=list)
    token_format: Literal["wikiann"] | None = None
    tag_format: Literal["indices"] | None = None
    tag_alignment: Literal["offsets", "byte_offsets"] | None = None
    token_alignment: Literal["offsets"] | None = None
    text_alignment: Literal["offsets"] | None = None


type Loader = HuggingfaceLoader | GitLoader | GenericLoader


@dataclass(config=_strict_config)
class DatasetPartition:
    """
    Configuration for a dataset partition.

    Attributes:
        type: Whether this is a subset or split.
        name: Name of the partition.
    """

    type: Literal["subset", "split"]
    name: str


@dataclass(config=_strict_config)
class Agreement:
    """
    Inter-annotator agreement metrics, if reported.

    Attributes:
        value: Agreement score (single value or per-label scores).
        metric: Name of the agreement metric used.
    """

    value: dict[str, float] | float
    metric: str | None = None


@dataclass(config=_strict_config)
class DataSource:
    """
    A source from which the original raw or annotated data was collected.

    Attributes:
        source: Name or identifier of the source.
        url: URL to the source.
    """

    source: str
    url: str


@dataclass(config=_strict_config)
class AnnotationMetadata:
    """
    Metadata about the annotation process.

    Attributes:
        type: Type of annotation (e.g., manual, heuristic).
        annotator_count: Number of annotators.
        features: Notable features of the annotation process.
        agreement: Inter-annotator agreement metrics reported for the
            data.
    """

    type: str | None = None
    annotator_count: int | str | None = None
    features: list[str] | None = None
    agreement: Agreement | list[Agreement] | None = None

    def merge(self, other: Self | None) -> Self:
        """
        Merges this annotation metadata with another instance.

        Args:
            other: Another AnnotationMetadata instance to merge with or
                None.

        Returns:
            A new AnnotationMetadata with values from `other` taking
            precedence.
        """

        if other is None:
            return self

        return self.__class__(
            self.type if other.type is None else other.type,
            self.annotator_count if other.annotator_count is None else other.annotator_count,
            self.features if other.features is None else other.features,
            self.agreement if other.agreement is None else other.agreement,
        )


@dataclass(config=_strict_config)
class Licenses:
    """
    License information for annotations and text.

    Attributes:
        annotations: License for annotations (string or per-source
            mapping).
        text: License for text content (string or per-source mapping).
    """

    annotations: str | dict[str, str]
    text: str | dict[str, str]

    def licenses(self) -> list[str]:
        """
        Collects all unique licenses in a sorted list

        Returns:
            All unique licenses in lexographically sorted order
        """
        return sorted(
            [
                *((self.annotations,) if isinstance(self.annotations, str) else self.annotations.values()),
                *((self.text,) if isinstance(self.text, str) else self.text.values()),
            ]
        )


type SentenceBoundaryType = Literal["full", "sections", "none"] | None


@dataclass(config=_strict_config, kw_only=True)
class Metadata:
    """
    General metadata for a dataset or subset.

    Attributes:
        license: License information (string or `Licenses` for separate
            text/annotation licenses).
        annotation: Annotation process metadata.
        primary_domain: Primary domain of the data (e.g., medical,
            legal).
        other_domains: Additional broad domains present in the data.
        finegrained_domains: Fine-grained domains present in the data.
        data_sources: List of data sources.
        dataset_lineage: Provenance information if the data was derived
            from on or multiple previously published datasets.
        label_set_standard: Standard or convention of the label set
            (such as OntoNotes or XBRL tags).
        document_boundaries: Whether the original documents or parts of
            documents can be restored from the data based on available
            boundary information or file structure.
        sentence_boundaries: Whether the data is segmented into
            sentences or sections.
    """

    license: str | Licenses | None = None
    annotation: AnnotationMetadata | None = None
    primary_domain: str | None = None
    other_domains: list[str] | None = None
    finegrained_domains: list[str] | None = None
    data_sources: list[str | DataSource] | None = None
    dataset_lineage: list[str] | None = None
    label_set_standard: str | None = None
    document_boundaries: Literal["full", "partial", "none"] | None = None
    sentence_boundaries: SentenceBoundaryType = None

    def licenses(self) -> list[str]:
        """
        Collects all unique licenses of the data in a sorted list

        Returns:
            All unique licenses in lexographically sorted order
        """
        match self.license:
            case None:
                return []
            case Licenses():
                return self.license.licenses()
            case str():
                return [self.license]

    def merge(self, other: "Metadata") -> "Metadata":
        """
        Merges this metadata with another instance.

        Args:
            other: Another Metadata instance to merge with.

        Returns:
            A new Metadata with values from other taking precedence
            where None.
        """

        return Metadata(
            annotation=other.annotation if self.annotation is None else self.annotation.merge(other.annotation),
            license=self.license if other.license is None else other.license,
            primary_domain=self.primary_domain if other.primary_domain is None else other.primary_domain,
            other_domains=self.other_domains if other.other_domains is None else other.other_domains,
            finegrained_domains=self.finegrained_domains
            if other.finegrained_domains is None
            else other.finegrained_domains,
            data_sources=self.data_sources if other.data_sources is None else other.data_sources,
            dataset_lineage=self.dataset_lineage if other.dataset_lineage is None else other.dataset_lineage,
            label_set_standard=self.label_set_standard
            if other.label_set_standard is None
            else other.label_set_standard,
            document_boundaries=self.document_boundaries
            if other.document_boundaries is None
            else other.document_boundaries,
            sentence_boundaries=self.sentence_boundaries
            if other.sentence_boundaries is None
            else other.sentence_boundaries,
        )


@dataclass(config=_strict_config)
class SplitSelector:
    """
    Selector for dataset splits containing the hierarchical subset path and an optional split name.

    Attributes:
        subset_hierarchy: Path through the subset hierarchy (list of
            subset names).
        split: Optional split name (train, validation, test, or None).
    """

    subset_hierarchy: list[str]
    split: str | None

    @classmethod
    def parse(cls, selector: str) -> Self:
        """
        Parses a selector string into a SplitSelector.

        Args:
            selector: Selector string in format "subset1.subset2.split".
                Wild cards can be used for matching any subset on the
                specified level of the hierarchy using an asterisk, such
                as "subset1.*.split"

        Returns:
            A new SplitSelector instance.
        """

        parts = selector.split(".")
        if len(parts) > 1 and parts[-1] in {"train", "validation", "test"}:
            split = parts[-1]
            parts = parts[:-1]
        else:
            split = None

        return cls([part for part in parts if part], split)

    def specificity(self) -> tuple[int, int]:
        """
        Calculates the specificity of this selector.

        Returns:
            Specificity of the selector in the form (1 or 0 indicating
            whether a specific split was specified, count of non-
            wildcard selectors).
        """

        return (int(self.split is not None), sum(int(selector != "*") for selector in self.subset_hierarchy))

    def matches(self, hierarchy: list[str], split: str) -> bool:
        """
        Checks if this selector matches the given hierarchy and split.

        Args:
            hierarchy: The subset hierarchy to check against.
            split: The split name to check against.

        Returns:
            True if this selector matches.
        """

        return (
            (self.split is None or self.split == split)
            and len(hierarchy) == len(self.subset_hierarchy)
            and all(selector in ("*", subset) for selector, subset in zip(self.subset_hierarchy, hierarchy))
        )


@dataclass(config=_strict_config)
class SubMetadata(Metadata):
    """
    Metadata that applies to all splits matching the given selectors.

    Attributes:
        license: License information (string or `Licenses` for separate
            text/annotation licenses).
        annotation: Annotation process metadata.
        primary_domain: Primary domain of the data (e.g., medical,
            legal).
        other_domains: Additional broad domains present in the data.
        finegrained_domains: Fine-grained domains present in the data.
        data_sources: List of data sources.
        dataset_lineage: Provenance information if the data was derived
            from on or multiple previously published datasets.
        label_set_standard: Standard or convention of the label set
            (such as OntoNotes or XBRL tags).
        document_boundaries: Whether the original documents or parts of
            documents can be restored from the data based on available
            boundary information or file structure.
        sentence_boundaries: Whether the data is segmented into
            sentences or sections.
        split: List of split selectors that this metadata applies to.
    """

    split: list[Annotated[SplitSelector, BeforeValidator(SplitSelector.parse)]]


@dataclass(config=_strict_config)
class MELDDataset:
    """
    Complete dataset definition, including metadata and preprocessing data pipes for integration into MELD.

    Attributes:
        citekeys: Citation keys for this dataset in the included BibTeX bibliography.
        source: Data loader configuration.
        format: Data format specification.
        metadata: List of dataset, subset and split-specific metadata
            that can be resolved via a CSS-style cascade to reduce
            repetition.
        settings: Any evaluation settings defined for the dataset (such
            as coarse-grained and fine-grained, few-shot, etc.) and
            which subsets or splits they include.
        note: Optional notes about this dataset.
        use_shared_cache: Whether to use a shared cache for downloaded
            resources in cases where multiple datasets are downloaded
            from the same source.
    """

    citekeys: list[str]
    source: Loader
    format: Format
    metadata: list[SubMetadata]
    settings: dict[str, list[DatasetPartition]] | None = None
    note: str | None = None
    use_shared_cache: str | None = None


type DatasetManifest = dict[str, MELDDataset]


def load_manifest(path_or_dict: PathLike | str | dict[str, Any] | None = None) -> DatasetManifest:
    """
    Loads and deserializes a dataset manifest from a file or dictionary.

    Args:
        path_or_dict: Path to a manifest JSON file, or a dictionary to
            validate.

    Returns:
        Parsed `DatasetManifest` instance.
    """

    validator = TypeAdapter(DatasetManifest)
    if path_or_dict is None:
        path_or_dict = {}
        for metadata_path in (resources.files("meld") / "package_data/datasets").iterdir():
            with metadata_path.open("r") as file:
                path_or_dict.update(json.load(file))

    if not isinstance(path_or_dict, dict):
        with Path(path_or_dict).open("r") as file:
            path_or_dict = json.load(file)

    return validator.validate_python(path_or_dict)


@dataclass(config=_strict_config)
class _CompactLabelMapEntry:
    """
    A compact label map entry defining label mappings for specific subsets and tagsets.

    Attributes:
        subsets: List of subset hierarchies this label map applies to.
        tagset: Name of the tagset this label map is for.
        label_map: Dictionary mapping source labels to target labels.
    """

    subsets: list[tuple[str, ...]]
    tagset: str
    label_map: dict[str, str]


type _CompactLabelMap = dict[str, list[_CompactLabelMapEntry]]


def _load_compact_label_map(path_or_dict: PathLike | str | dict[str, Any] | None = None) -> _CompactLabelMap:
    """
    Loads a compact label map from a file or dictionary. Loads the included normalized label mapping if `path_or_dict` is `None`.

    Args:
        path_or_dict: Path to a label map JSON file, or a dictionary to validate. If `None`, loads the included normalized label mapping.

    Returns:
        Parsed `_CompactLabelMap` instance.
    """
    validator = TypeAdapter(_CompactLabelMap)
    if path_or_dict is None:
        path_or_dict = {}
        with (resources.files("meld") / "package_data/label_map.json").open("r") as file:
            return validator.validate_json(file.read())

    if isinstance(path_or_dict, dict):
        return validator.validate_python(path_or_dict)

    with Path(path_or_dict).open("r") as file:
        return validator.validate_json(file.read())


type LabelMap = dict[str, dict[tuple[str, ...], dict[str, dict[str, str]]]]


def load_label_map(path_or_dict: PathLike | str | dict[str, Any] | None = None) -> LabelMap:
    """
    Loads and deserializes a label map from a file or dictionary. Loads the included normalized label mapping if `path_or_dict` is `None`.

    Args:
        path_or_dict: Path to a label map JSON file, or a dictionary to validate. If `None`, loads the included normalized label mapping.

    Returns:
        Parsed nested dictionary containing a mapping for each dataset, subset, and tagset.
    """
    label_map = _load_compact_label_map(path_or_dict)

    expanded_label_maps = {}
    for dataset, entries in label_map.items():
        expanded_label_maps[dataset] = dataset_label_map = {}
        for entry in entries:
            for subset in entry.subsets:
                if (subset_maps := dataset_label_map.get(subset)) is None:
                    dataset_label_map[subset] = subset_maps = {}

                subset_maps[entry.tagset] = entry.label_map

    return expanded_label_maps
