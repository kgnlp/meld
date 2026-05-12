"""Readers for NER dataset formats."""

import ast
import csv
import dataclasses
import itertools
import json
import logging
import operator
import re
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

import polars as pl
from datasets import Dataset
from lxml import etree

from meld import conll
from meld.conll import (
    CoNLLColumnarRegistry,
    CoNLLDialectRegistry,
    CoNLLMetadata,
    CoNLLStackOverflowNER,
    NerSuiteCoNLL,
    RowParser,
    Sentence,
    space_separated_segments,
)
from meld.document import BIO, Annotation, LabeledText, LabeledTokens, NERDocument, Span
from meld.formats import (
    DEFAULT_TAGSET,
    default_tagset,
)
from meld.manifest import (
    ByteOffsetJSONLArguments,
    CoNLLArguments,
    DatasetConvertStep,
    DetokenizerType,
    EBMNLPStandoffArguments,
    OffsetCSVArguments,
    PlainSpanArguments,
    ReaderConfiguration,
    SofcStandoffArguments,
    StandoffArguments,
)
from meld.registry import Registry
from meld.tokenization import Detokenizer, align_tokens_with_text, bio_to_spans

logger = logging.getLogger("meld")


@dataclass(slots=True)
class FileSource:
    """
    Represents a file source for reading lines from a text file.

    Attributes:
        get_lines: A callable that yields lines from the file.
        path: The Path object pointing to the file.
    """

    path: Path

    def lines(self) -> Iterable[str]:
        """
        Opens an iterator over the file's lines.

        Yields:
            Lines from the file.
        """

        with self.path.open("r", encoding="utf-8") as file:
            yield from file

    def read(self) -> str:
        """
        Reads the full text content of the file into a string.

        Yields:
            The text content of the file.
        """

        with self.path.open("r", encoding="utf-8") as file:
            return file.read()


@dataclass(slots=True)
class DatasetSource:
    """
    Represents a Huggingface Dataset source for NER data.

    Attributes:
        dataset: The Huggingface Dataset instance.
        text_column: Name of the column containing text tokens.
        default_tag_column: Name of the default NER tag column.
    """

    dataset: Dataset
    text_column: str
    default_tag_column: str


type DataSource = DatasetSource | FileSource | dict[str, list[FileSource]]


@dataclass(slots=True)
class SubsetName:
    """
    Represents a dataset subset with an optional type qualifier.

    Attributes:
        name: Name of the subset.
        type: Optional type qualifier (e.g., "language", "year").
    """

    name: str
    type: str | None = None


@dataclass(slots=True)
class SplitData:
    """
    Represents a dataset split with its data sources and metadata.

    Attributes:
        name: Name of the split.
        data: List of data sources for this split.
        language: Language code for the split's content.
        subset: List of Subset instances defining the subset hierarchy.
        metadata: Additional metadata dictionary.
    """

    name: str
    data: list[DataSource]
    language: str
    subset: list[SubsetName] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def slug(self) -> str:
        """
        Generates a slug identifier for the split based on its subset hierarchy.

        Returns:
            A string slug combining subset names and qualifiers.
        """

        return "--".join(
            subset.name if subset.type is None else f"{subset.type}-{subset.name}" for subset in self.subset
        )


type SplitPaths = list[Path | dict[str, list[Path]]]


@dataclass(slots=True)
class NamedTagSet:
    """
    Represents a named set of NER tags with optional BIO label mapping.

    Attributes:
        name: Name of the tagset.
        column: Optional column name for this tagset in datasets.
        label_map: Optional mapping from indices to BIO labels.
    """

    name: str = DEFAULT_TAGSET
    column: str | None = None
    label_map: dict[int, BIO] | None = None


class CoNLLPreprocessorRegistry(Registry[Callable[[Iterable[str]], Iterator[str]]]):
    """Registry for CoNLL preprocessor functions."""


@CoNLLPreprocessorRegistry.register("stackoverflow_ner")
def _stackoverflow_ner_preprocessor(lines: Iterable[str]) -> Iterator[str]:
    """
    Preprocessor for StackOverflowNER dataset format.

    Handles special markers for questions and answers, and removes code block annotations from which the original code was removed.

    Args:
        lines: Iterable of input lines.
    Yields:
        Preprocessed lines with proper document segmentation.
    """

    for segment in conll.space_separated_segments(lines, enforce_blank_lines=False):
        first_line = segment[0]
        # Ignore metadata and omitted code block and output block placeholders
        if re.match(r"^(Question_URL|OP_BLOCK|Issue_Event_Link)", first_line) or any(
            re.match(r"^(CODE_BLOCK|names\.CODE_BLOCK)", line) for line in segment
        ):
            yield ""
        # Replace start segments with docstart
        elif re.match(r"^(Answer_to_Question_ID|Question_ID|Repository_Name)", first_line):
            # Docstart segment separated with a blank line
            yield CoNLLStackOverflowNER.docstart
            yield ""
        else:
            # Yield full segment with final blank line if no special handling is needed
            yield from segment
            yield ""


@CoNLLPreprocessorRegistry.register("e-ner")
def _e_ner_preprocessor(lines: Iterable[str]) -> Iterator[str]:
    """
    Preprocessor for the E-NER dataset.

    Fixes erroneous "P" tags and standardizes "LOC" labels to "LOCATION".

    Args:
        lines: Iterable of CSV input lines.
    Yields:
        Preprocessed tab-separated lines.
    """

    p_counter = 0
    loc_counter = 0
    for columns in csv.reader(lines):
        if not columns:
            yield ""
            continue

        form, label = columns
        if not form and label == "O":
            # Interpret empty word forms as blank lines
            yield ""
        elif label == "P":
            # Replace erroneous P tags with O (https://github.com/terenceau1/E-NER-Dataset/issues/1)
            p_counter += 1
            yield f"{form}\tO"
        elif label.endswith("LOC"):
            loc_counter += 1
            yield f"{form}\t{label.split('-')[0]}-LOCATION"
        else:
            # Convert CSV format to tab separated rows
            yield f"{form}\t{label}"

    if p_counter:
        logger.warning(f'Fixed {p_counter} erroneous "P" tags')
    if loc_counter:
        logger.warning(f'Standardized {loc_counter} instances of "LOC" to "LOCATION" tags')


_PIONER_FILTER = {"<li>", "</li>", "<center>", "</poem>", "<ref>", "</ref>", "<![CDATA[", "//", "\ufeff"}


@CoNLLPreprocessorRegistry.register("pioner")
def _pioner_preprocessor(lines: Iterable[str]) -> Iterator[str]:
    """
    Preprocessor for the PIONER dataset.

    Filters out lines with spurious markup tokens and byte-order marks, and removes spurious <nowiki/> tags from tokens.

    Args:
        lines: Iterable of input lines.
    Yields:
        Cleaned lines with markup removed.
    """

    for line in lines:
        # Filter most spurious markup tokens in the data
        if line.split(" ", 1)[0] in _PIONER_FILTER:
            continue
        # Remove spurious <nowiki/> and leading <ref> tags from the start of lines in the development and train set
        # NOTE: Currently doesn't remove more complex <ref markup
        yield re.sub(r"(?:<nowiki/>|^<ref>)", "", line)


@CoNLLPreprocessorRegistry.register("nytk_nerkor")
def _nytk_nerkor_preprocessor(lines: Iterable[str]) -> Iterator[str]:
    """
    Preprocessor for the NyTK Nerkor dataset.

    Fixes invalid BIO tags by replacing them with "O".

    Args:
        lines: Iterable of tab-separated input lines.
    Yields:
        Lines with corrected BIO tags.
    """

    for line in lines:
        if not line.strip() or line[0] == "#":
            yield line
            continue

        prefix, tag, lemma = line.rsplit("\t", 2)

        if tag[0] not in {"B", "I", "O"}:
            logger.warning(f'Replaced invalid BIO string "{tag[0]}" with "O"')
            yield f"{prefix}\tO\t{lemma}\n"
        else:
            yield line


class NERReader(Protocol):
    def read_split(self, split: SplitData) -> Iterable[NERDocument]: ...


class ReaderRegistry(Registry[type[NERReader]]):
    """Registry for CoNLL preprocessor functions."""


def _to_labeled_tokens(
    sentence: Sentence, tagsets: list[NamedTagSet] | None = None, convert_bioes: bool = False
) -> LabeledTokens:
    """
    Converts a single CoNLL Sentence to a LabeledTokens instance.

    Args:
        sentence: The CoNLL Sentence to convert.
        tagsets: Optional list of NamedTagSet instances for tag set
            configuration.
        convert_bioes: Whether to convert BIOES tags to BIO format.

    Returns:
        A LabeledTokens instance with the tokens and labels extracted
        from the sentence.

    Raises:
        ValueError: If multiple tagsets are provided which are currently
            unsupported.
    """

    if tagsets is None:
        label_map = None
    else:
        try:
            [tagset] = tagsets
            label_map = tagset.label_map
        except ValueError:
            raise ValueError("Multiple tagsets are currently unsupported by the CoNLL reader")

    if label_map is None:
        labels = (row.ner for row in sentence.rows)
    else:
        labels = (label_map[row.ner] for row in sentence.rows)

    if convert_bioes:
        labels = (tag.bioes_to_bio() for tag in labels)

    return LabeledTokens(
        [row.form for row in sentence.rows],
        default_tagset(list(labels)),
    )


def _get_sentence_text(metadata: CoNLLMetadata | None) -> str:
    """
    Extracts the untokenized reference text of a sentence from CoNLL metadata.

    Args:
        metadata: CoNLLMetadata containing sentence text or `None`.

    Returns:
        The sentence text string.

    Raises:
        ValueError: If metadata is None.
    """

    if metadata is None:
        raise ValueError("Encountered sentence in a commented CoNLL file without a comment")

    return metadata["text"]


def _detokenize_nersuite_offsets(rows: list[NerSuiteCoNLL]) -> tuple[str, list[Span]]:
    """
    Reconstructs the untokenized text and spans from NERSuite-style offset-annotated tokens.
    Assumes single spaces between sentences of a document since e.g. newlines cannot be reconstructed without the original text as a reference.

    Args:
        rows: List of NerSuiteCoNLL token rows.

    Returns:
        Tuple of (reconstructed text, list of Span instances).
    """

    # Starts relative to the current sequence if nersuite offsets are relative to the full document
    last_end = 0
    document_offset = rows[0].start
    segments = []
    spans = []
    for row in rows:
        start = row.start - document_offset
        end = row.end - document_offset
        # Fill gaps between the offsets with spaces
        segments.append((start - last_end) * " ")
        segments.append(row.form)
        last_end = end
        spans.append(Span(start, end))

    return "".join(segments), spans


@ReaderRegistry.register("conll")
@dataclass(slots=True)
class CoNLLReader:
    """
    Reader for CoNLL-formatted NER datasets.

    Attributes:
        arguments: CoNLLArguments configuration for reading.
        detokenizer_type: Type of detokenization to apply for text
            reconstruction.
        tagsets: Optional list of NamedTagSet instances for tag
            configuration.
    """

    arguments: CoNLLArguments
    detokenizer_type: DetokenizerType
    tagsets: list[NamedTagSet] | None = None

    def __post_init__(self) -> None:
        """Initializes tagsets from a label map if provided."""
        # Invert label map for index to BIO conversion
        self.tagsets = (
            None
            if self.arguments.label_map is None
            else [
                NamedTagSet(tagset, label_map={index: label for label, index in label_map.items()})
                for tagset, label_map in self.arguments.label_map.items()
            ]
        )

    def _conll_to_document(self, sentences: list[Sentence], detokenizer: Detokenizer) -> NERDocument:
        """
        Converts a list of CoNLL sentences to a NERDocument.

        Args:
            sentences: List of Sentence instances.
            detokenizer: Detokenizer for text reconstruction.

        Returns:
            A NERDocument with detokenized text and labeled tokens.
        """

        first_meta = sentences[0].meta
        return detokenizer.tokens_to_document(
            [
                _to_labeled_tokens(
                    sentence,
                    self.tagsets,
                    self.arguments.bioes_to_bio,
                )
                for sentence in sentences
            ],
            None
            if first_meta is None or "text" not in first_meta
            else [_get_sentence_text(sentence.meta) for sentence in sentences],
        )

    def _nersuite_to_document(self, sentences: list[Sentence[NerSuiteCoNLL]], _detokenizer: Detokenizer) -> NERDocument:
        """
        Converts NERSuite-style sentences to a NERDocument with proper offset handling.

        Args:
            sentences: List of NERSuite Sentence instances.

        Returns:
            A NERDocument with text, offsets, and labeled tokens.
        """

        labeled_text = []
        labeled_tokens = []
        last_end = 0

        for sentence in sentences:
            pre_tokenized = _to_labeled_tokens(
                sentence,
                self.tagsets,
                self.arguments.bioes_to_bio,
            )
            labeled_tokens.append(pre_tokenized)
            text, spans = _detokenize_nersuite_offsets(sentence.rows)

            if labeled_text:
                labeled_text[-1].space_after = (sentence.rows[0].start - last_end) * " "

            last_end = sentence.rows[-1].end

            labeled_text.append(
                LabeledText(text, {tagset: bio_to_spans(tags, spans) for tagset, tags in pre_tokenized.labels.items()})
            )

        return NERDocument(labeled_text, labeled_tokens)

    def _read_file_shard(
        self, source: FileSource, detokenizer: Detokenizer, metadata: dict[str, Any], shards_are_documents: bool = False
    ) -> Iterable[NERDocument]:
        """
        Reads and processes a file shard into `NERDocument`s.

        Args:
            source: FileSource containing the input file.
            detokenizer: Detokenizer for text reconstruction.
            metadata: Processing metadata for the file.
            shards_are_documents: Whether each shard represents a
                complete document.
        Yields:
            NERDocument instances.

        Raises:
            KeyError: If the CoNLL dialect is unsupported.
        """

        try:
            dialect = CoNLLDialectRegistry.get(self.arguments.dialect)
        except KeyError:
            raise KeyError(f"Unsupported dialect for CoNLL files: {self.arguments.dialect}")

        parse_arguments = metadata.get("parse_arguments") or {}

        # Select dialect specific CoNLL row to NERDocument processor
        process_document = self._nersuite_to_document if dialect == NerSuiteCoNLL else self._conll_to_document

        lines = source.lines()
        if self.arguments.preprocessor is not None:
            lines = CoNLLPreprocessorRegistry.get(self.arguments.preprocessor)(lines)

        sentences: list[Sentence] = []
        for document in conll.parse(
            lines, dialect, self.arguments.delimiter, self.arguments.enforce_blank_lines, **parse_arguments
        ):
            if shards_are_documents:
                # Assume no nested documents if the "shards_are_documents" option is enabled
                try:
                    [sentence] = document
                    sentences.append(sentence)
                except ValueError:
                    raise ValueError(
                        'Expected exactly one sentence in document when "shards_are_documents" is True, '
                        f"but got {len(document)} sentences."
                    )
            else:
                # Type of document at this point can currently not be inferred
                yield process_document(document, detokenizer)  # pyright: ignore

        if sentences:
            yield process_document(sentences, detokenizer)

    def _read_dataset_shard(
        self, source: DatasetSource, detokenizer: Detokenizer, shards_are_documents: bool = False
    ) -> Iterable[NERDocument]:
        """
        Reads and processes a HuggingFace dataset shard into NERDocuments.

        Args:
            source: DatasetSource containing the dataset.
            detokenizer: Detokenizer for text reconstruction.
            shards_are_documents: Whether each shard represents a
                complete document.
        Yields:
            NERDocument instances.

        Raises:
            KeyError: If the dialect is unsupported.
            ValueError: If a preprocessor is specified which is not
                supported by CoNLL-style columnar datasets.
            ValueError: If multiple tagsets are provided which are
                currently unsupported.
        """

        try:
            dialect = CoNLLColumnarRegistry.get(self.arguments.dialect)
        except KeyError:
            raise KeyError(f"Unsupported dialect for CoNLL-style columnar data: {self.arguments.dialect}")

        if self.arguments.preprocessor is not None:
            raise ValueError(
                f"Preprocessor {self.arguments.preprocessor} not supported for CoNLL-style columnar datasets"
            )

        if self.tagsets is None:
            tag_column = source.default_tag_column
        else:
            try:
                [tagset] = self.tagsets
                tag_column = tagset.column or source.default_tag_column
            except ValueError:
                raise ValueError("Multiple tagsets are currently unsupported for CoNLL-style datasets")

        dataset = source.dataset.rename_columns({source.text_column: "form", tag_column: "ner"})
        previous_document_index = None

        # Pick source specific label map if possible and fall back to the global mapping
        sentences = []
        for sentence in conll.parse_columns(dataset.to_iterable_dataset(), dialect):
            if (
                not shards_are_documents
                and previous_document_index != (doc_idx := sentence.rows[0].doc_idx)
                and sentences
            ):
                previous_document_index = doc_idx
                yield detokenizer.tokens_to_document(sentences)
                sentences = []

            sentences.append(_to_labeled_tokens(sentence, self.tagsets, self.arguments.bioes_to_bio))

        if sentences:
            yield detokenizer.tokens_to_document(sentences)

    def read_split(self, split: SplitData) -> Iterable[NERDocument]:
        """
        Reads a split and yields NERDocuments for each shard.

        Args:
            split: Split instance with data sources.
        Yields:
            NERDocument instances from reading each shard.

        Raises:
            ValueError: If a shard has an unsupported type.
        """

        for shard in split.data:
            detokenizer = Detokenizer(split.language, self.detokenizer_type)
            if isinstance(shard, FileSource):
                yield from self._read_file_shard(
                    shard, detokenizer, split.metadata, self.arguments.shards_are_documents
                )
            elif isinstance(shard, DatasetSource):
                yield from self._read_dataset_shard(shard, detokenizer, self.arguments.shards_are_documents)
            else:
                raise TypeError(f"Encountered shard with unsupported type: {shard}")


class LineReader(metaclass=ABCMeta):
    """Abstract base class for readers that process text line-by-line."""

    @abstractmethod
    def _read_lines(self, lines: FileSource) -> Iterable[NERDocument]:
        """
        Reads and processes lines from a FileSource.

        Args:
            lines: FileSource to read from.
        Yields:
            NERDocument instances.
        """

    def read_split(self, split: SplitData) -> Iterable[NERDocument]:
        """
        Reads splits and yields NERDocuments.

        Args:
            split: Split instance with data sources.
        Yields:
            NERDocument instances.

        Raises:
            ValueError: If a shard is not a text file.
        """

        for shard in split.data:
            if not isinstance(shard, FileSource):
                raise TypeError(
                    f"{self.__class__.__name__} only supports text files, but found shard of type {type(shard)}"
                )

            yield from self._read_lines(shard)


def _fix_mismatched_offset(text: str, expected: str, from_offsets: str, spans: Span, max_deviation: int = 5) -> Span:
    """
    Attempts to fix mismatched annotation offsets by searching for the expected text within a deviation range.

    Args:
        text: The full text containing the annotation.
        expected: The expected annotation text.
        from_offsets: The actual text found at the given offsets for
            logging and error reporting.
        spans: Current span offsets.
        max_deviation: Maximum character deviation to search for correct
            offset.

    Returns:
        Fixed Span with corrected offsets.

    Raises:
        ValueError: If the offset mismatch cannot be automatically
            fixed.
    """

    for deviation, direction in itertools.product(range(1, max_deviation + 1), (-1, 1)):
        current_deviation = direction * deviation
        # Note: Does not fix a mismatch in the stop offset
        candidate_span = text[spans.start + current_deviation : spans.stop + current_deviation]
        if candidate_span == expected:
            logger.warning(
                f"Automatically fixed mismatched text offset: {expected!r} with original offsets {spans} was off by {current_deviation} and matched {from_offsets!r} instead"
            )
            return Span(spans.start + current_deviation, spans.stop + current_deviation)

    raise ValueError(
        f"Unrecoverable annotation offset mismatch. Expected {expected!r} but got {from_offsets!r} at offsets {spans}"
    )


@ReaderRegistry.register("bioc_xml")
@dataclass(slots=True)
class BioCXMLReader(LineReader):
    """Reader for NER datasets in the BioC XML format."""

    def _read_lines(self, lines: FileSource) -> Iterable[NERDocument]:
        """
        Parses BioC XML and yields NERDocuments with annotations, including any discontinuous spans.

        Args:
            lines: FileSource containing BioC XML.
        Yields:
            NERDocument instances with passage-level annotations.
        """

        document = etree.fromstringlist(lines.lines())
        document_ids = set()
        for document in document.xpath("/collection/document"):
            [document_id] = document.xpath("./id/text()")
            if document_id in document_ids:
                logger.warning(f"Skipping duplicate document ID {document_id} in file")
                continue

            document_ids.add(document_id)

            passages = []
            for passage in document.xpath("./passage"):
                [text] = passage.xpath("./text/text()")
                text = text.strip()
                [passage_offset] = passage.xpath("./offset/text()")
                passage_offset = int(passage_offset)
                labels = []

                for annotation in passage.xpath("./annotation"):
                    # Each annotation may have one or multiple locations for discontinuous spans
                    offsets = tuple(
                        Span.from_run_length(int(location.get("offset")) - passage_offset, int(location.get("length")))
                        for location in annotation.xpath("./location")
                    )
                    [label] = annotation.xpath('./infon[@key="type"]/text()')
                    [annotation_text] = annotation.xpath("./text/text()")

                    found_subspans = " ".join(text[subspan.start : subspan.stop] for subspan in offsets)
                    # If the given annotation text and the given start offset of the span in the text don't match, attempt to match them up and fix them automatically
                    # Note: This approach does not handle cases where the length of the span is also incorrect
                    if found_subspans != annotation_text:
                        fixed_offsets = []
                        expected_tokens = re.split("( )", annotation_text)
                        token_offset = 0
                        for subspan in offsets:
                            length = 0
                            for token in expected_tokens:
                                length += len(token)
                                token_offset += 1
                                if length >= len(subspan):
                                    break

                            found = text[subspan.start : subspan.stop]
                            expected = "".join(expected_tokens[:token_offset]).strip()
                            if found == expected:
                                fixed_offsets.append(subspan)
                            else:
                                fixed_offsets.append(_fix_mismatched_offset(text, expected, found, subspan))
                            expected_tokens = expected_tokens[token_offset:]

                    labels.append(Annotation(label, offsets))

                # NOTE: Uses newline delimiters since passages are usually the title and abstract of an article
                passages.append(LabeledText(text, default_tagset(labels), "passage", space_after="\n"))

            # No space after the last passage in the document
            passages[-1].space_after = ""
            yield NERDocument(passages)


def _check_span(expected_span: str, text: str, start: int, end: int):
    """
    Validates that a span matches the expected text and raises an exception if span lengths differ.

    Args:
        expected_span: Expected text at the given span.
        text: Full text containing the span.
        start: Start offset of the span.
        end: End offset of the span.

    Raises:
        ValueError: If the span doesn't match the expected text and the
            lengths of the spans differ.
    """

    actual_span = text[start:end]

    if expected_span != actual_span:
        if len(expected_span) != len(actual_span):
            raise ValueError(
                f"Annotation offset mismatch. Expected {expected_span!r} but got {actual_span!r} at offset ({start}, {end})"
            )

        logger.warning(f"Annotation inconsistency: Expected {expected_span!r}, found {actual_span!r}")


@ReaderRegistry.register("offset_csv")
@dataclass(slots=True)
class OffsetCSVReader(LineReader):
    """
    Reader for OffsetCSV format with JSON-encoded span annotations.

    Attributes:
        arguments: Configuration for the OffsetCSV format reader.
    """

    arguments: OffsetCSVArguments

    def _read_lines(self, lines: FileSource) -> Iterable[NERDocument]:
        """
        Reads OffsetCSV files with JSON-encoded span annotations.

        Args:
            lines: FileSource containing the CSV file.
        Yields:
            NERDocument instances with passages and annotations.
        """

        data = (
            pl.scan_csv(lines.path)
            .select([self.arguments.text_column, self.arguments.offsets_column])
            .with_columns(
                pl.col(self.arguments.offsets_column).str.json_decode(
                    dtype=pl.List(
                        pl.Struct({"start": pl.Int64, "end": pl.Int64, "type": pl.String(), "entity": pl.String()})
                    )
                )
            )
        ).collect()

        for text, offsets in data.iter_rows():
            annotations = []
            for offset in offsets:
                start = offset["start"]
                end = offset["end"]

                _check_span(offset["entity"], text, start, end)

                annotations.append(Annotation(offset["type"], (Span(start, end),)))

            yield NERDocument([LabeledText(text, default_tagset(annotations), "passage")])


@ReaderRegistry.register("byte_offset_jsonl")
@dataclass(slots=True)
class ByteOffsetJSONLReader(LineReader):
    """
    Reader for the ByteOffsetJSONL format.

    Attributes:
        arguments: Configuration for the ByteOffsetJSONL format reader.
    """

    arguments: ByteOffsetJSONLArguments

    def _read_lines(self, lines: FileSource) -> Iterable[NERDocument]:
        """
        Reads ByteOffsetJSONL files with byte-level annotations.

        Args:
            lines: FileSource containing JSONL files.

        Yields:
            NERDocument instances with byte-aligned annotations.

        Raises:
            ValueError: If an annotation offset mismatch is detected.
        """

        text_key = self.arguments.text_key
        offsets_key = self.arguments.offsets_key
        target_key = self.arguments.annotated_span_target_key

        for line in lines.lines():
            sample = json.loads(line)
            # Treat as byte string
            text = sample[text_key]
            text_bytes = text.encode("utf-8")
            annotations = []

            if target_key:
                targets = sample[target_key]
                # Targets field of the form "PER: ... $$ ORG: ..."
                expected_spans = (
                    [target.split(":", 1)[1].strip() for target in targets.split("$$")] if targets else None
                )
            else:
                expected_spans = None

            # Track the location within the string in alignment with the byte string
            string_offset = 0
            # Track the position after the previous span in the byte string
            previous_byte_end = 0
            for i, offset in enumerate(sample[offsets_key]):
                start = offset["start_byte"]
                end = offset["limit_byte"]

                # Add the number of characters of text between spans
                string_offset += len(text_bytes[previous_byte_end:start].decode("utf-8"))
                # Track the start and end offsets of the span in the string
                string_start = string_offset
                string_offset += len(text_bytes[start:end].decode("utf-8"))

                previous_byte_end = end

                if (
                    expected_spans is not None
                    and (current_span := text[string_start:string_offset]) != expected_spans[i]
                ):
                    raise ValueError(
                        f"Annotation offset mismatch. Expected {expected_spans[i]!r} but got {current_span!r} at offset ({start}, {end})/({string_start}, {string_offset})"
                    )

                annotations.append(Annotation(offset["label"], (Span(string_start, string_offset),)))

            yield NERDocument([LabeledText(text, default_tagset(annotations))])


@ReaderRegistry.register("scirex_jsonl")
class SciRexJSONLReader(LineReader):
    """Reader for the SciREX JSONL format coontaining pre-tokenized text and NER labels."""

    def _read_lines(self, lines: FileSource) -> Iterable[NERDocument]:
        """
        Reads a file in SciREX JSONL format with sentence-level annotations.

        Note that this reader will *merge sentences* if span annotations would otherwise cross sentence boundaries.

        Args:
            lines: FileSource containing JSONL files.
        Yields:
            NERDocument instances with sentences and entity annotations.

        Raises:
            ValueError: If a span mismatch is detected.
            ValueError: If some labels have not been matched to the text.
        """

        for line in lines.lines():
            sample = json.loads(line)
            document = sample["words"]
            # NER labels in SciREX are not sorted
            ner_labels = sorted(sample["ner"], key=operator.itemgetter(0))
            sentences = []
            sentence_spans = sample["sentences"]
            ner_index = 0
            sentence_offset = 0
            current_sentence = 0

            while current_sentence < len(sentence_spans):
                start, end = sentence_spans[current_sentence]
                sentence_words = document[start:end]
                sentence = " ".join(sentence_words)
                # Offset in the original string accounting for spaces
                string_offsets = list(itertools.accumulate(len(word) + 1 for word in sentence_words))
                string_offsets.insert(0, 0)

                annotations = []
                for _ in range(ner_index, len(ner_labels)):
                    a_start, a_end, label = ner_labels[ner_index]
                    if a_start >= end:
                        break

                    expected = " ".join(document[a_start:a_end])
                    sentence_extended = False

                    # Note: If an entity annotations spans across multiple sentences, merge sentences until it can fit
                    while a_end > end:
                        sentence_extended = True
                        current_sentence += 1
                        _, end = sentence_spans[current_sentence]

                    if sentence_extended:
                        sentence_words = document[start:end]
                        sentence = " ".join(sentence_words)
                        string_offsets = list(itertools.accumulate(len(word) + 1 for word in sentence_words))
                        string_offsets.insert(0, 0)

                    a_start -= sentence_offset
                    a_end -= sentence_offset

                    string_start = string_offsets[a_start]
                    string_end = string_offsets[a_end] - 1

                    actual = sentence[string_start:string_end]
                    if actual != expected:
                        raise ValueError(f"Span mismatch {actual!r} : {expected!r}")

                    annotations.append(Annotation(label, (Span(string_start, string_end),)))
                    ner_index += 1

                sentence_offset += len(sentence_words)
                current_sentence += 1

                sentences.append(LabeledText(sentence, default_tagset(annotations), space_after=" "))

            # Sanity check ensuring all labels have been extracted
            if ner_index != len(ner_labels):
                raise ValueError(
                    f"Some labels have not been matched to the text. Expected {len(ner_labels)} labels, matched {ner_index}"
                )

            # Remove trailing space after last sentence in the document
            sentences[-1].space_after = ""
            yield NERDocument(sentences)


@ReaderRegistry.register("plain_spans")
@dataclass(slots=True)
class PlainSpanReader(LineReader):
    """
    Reader for an annotation format where lines of text followed are followed by JSON or Python dictionary formatted span-level annotations.

    Attributes:
        arguments: Configuration arguments for the plain span format.
    """

    arguments: PlainSpanArguments

    def _read_lines(self, lines: FileSource) -> Iterable[NERDocument]:
        """
        Reads plain span format files with text and span annotations.

        Args:
            lines: FileSource containing text and span annotations.

        Yields:
            NERDocument instances containing annotated passages.

        Raises:
            ValueError: If a span annotation mismatch is detected.
        """

        line_iterator = iter(lines.lines())
        for line in line_iterator:
            line = line.strip()
            # Ignore blank lines and tweet IDs
            if not line or line.startswith("#tid:"):
                continue

            text = line.strip()
            annotations = next(line_iterator).strip()

            # Handle multiline text by joining it and preserving newlines until an annotation is encountered
            # NOTE: Only occurs in the training data of DanfeNER
            while not annotations.startswith("["):
                text += "\n" + annotations
                annotations = next(line_iterator).strip()

            match self.arguments.span_format:
                case "json":
                    annotations = json.loads(annotations)
                case "python":
                    annotations = ast.literal_eval(annotations)

            span_annotations = []
            for start, end, label, text_span in annotations:
                if text[start:end] != text_span:
                    raise ValueError(f"Span annotation mismatch. Expected {text_span!r}, got {text[start:end]!r}")
                span_annotations.append(Annotation(label, (Span(start, end),)))

            yield NERDocument([LabeledText(text, default_tagset(span_annotations))])


class DetokenizingLineReader(metaclass=ABCMeta):
    """Abstract base class for readers that require detokenization of tokenized text."""

    @abstractmethod
    def _read_lines(self, lines: FileSource, detokenizer: Detokenizer) -> Iterable[NERDocument]:
        """
        Reads and detokenizes text from a file.

        Args:
            lines: FileSource containing tokenized text.
            detokenizer: Detokenizer for text reconstruction.

        Yields:
            NERDocument instances with detokenized text.
        """

    def read_split(self, split: SplitData) -> Iterable[NERDocument]:
        """
        Reads splits and yields documents after detokenization.

        Args:
            split: Split instance with data sources.

        Yields:
            NERDocument instances.

        Raises:
            ValueError: If a shard is not a text file.
        """

        detokenizer = Detokenizer(split.language)
        for shard in split.data:
            if not isinstance(shard, FileSource):
                raise TypeError(
                    f"{self.__class__.__name__} only supports text files, but found shard of type {type(shard)}"
                )

            yield from self._read_lines(shard, detokenizer)


@ReaderRegistry.register("scier_jsonl")
class SciERJSONLReader(DetokenizingLineReader):
    """
    Reader for the SciER JSONL format containing pre-tokenized text and NER labels provided as token indices.

    Each file contains multiple documents with sentence-level token indices for NER annotations.
    """

    def _read_lines(self, lines: FileSource, detokenizer: Detokenizer) -> Iterable[NERDocument]:
        """
        Reads SciER JSONL format and yields NERDocuments with detokenized text.

        Args:
            lines: FileSource containing JSONL files with pre-tokenized sentences.
            detokenizer: Detokenizer for reconstructing text from tokens.

        Yields:
            NERDocument instances with detokenized text and entity annotations.

        Raises:
            ValueError: If the number of sentences and label sequences doesn't match.
        """

        for line in lines.lines():
            sample = json.loads(line)
            token_offset = 0

            annotated_sentences = []
            sentences = sample["sentences"]
            ner = sample["ner"]

            # Sanity check
            if len(sentences) != len(ner):
                raise ValueError(
                    f"Number of sentences and label sequences doesn't match. Was {len(sentences)} and {len(ner)}"
                )

            for sentence, labels in zip(sentences, ner):
                text, spans = detokenizer.detokenize(sentence)

                annotated_sentences.append(
                    LabeledText(
                        text,
                        default_tagset(
                            [
                                Annotation(
                                    label, (Span(spans[start - token_offset].start, spans[end - token_offset].stop),)
                                )
                                for start, end, label in labels
                            ]
                        ),
                        space_after=" ",
                    )
                )

                token_offset += len(sentence)

            # Remove trailing space after last sentence in the document
            annotated_sentences[-1].space_after = ""
            yield NERDocument(annotated_sentences)


def _flatten_label(label: str | list[str]) -> str:
    """
    Flattens a label that may be a single string or a single-item list.

    Args:
        label: A string or single-item list containing the label.

    Returns:
        The flattened string label.
    """

    if isinstance(label, list):
        [label] = label

    return label


@ReaderRegistry.register("arabic_cross_dialectal_json")
class ArabicCrossDialectalJSONReader(DetokenizingLineReader):
    """Reader for the ArabicCrossDialectal JSON format with detokenization support."""

    def _read_lines(self, lines: FileSource, detokenizer: Detokenizer) -> Iterable[NERDocument]:
        """
        Reads lines in the ArabicCrossDialectal JSON format and parses them into NERDocuments.

        Args:
            lines: FileSource containing JSON data with tokens and NER
                annotations.
            detokenizer: Detokenizer for reconstructing the source text.

        Yields:
            NERDocument instances with detokenized text and entity annotations.
        """

        for sample in json.loads(lines.read()):
            tokens = sample["tokens"]
            text, spans = detokenizer.detokenize(tokens)

            yield NERDocument(
                [
                    LabeledText(
                        text,
                        default_tagset(
                            [
                                Annotation(_flatten_label(label), (Span(spans[start].start, spans[end].stop),))
                                for start, end, label in sample["ner"]
                            ]
                        ),
                    )
                ]
            )


@ReaderRegistry.register("standoff")
@dataclass(slots=True)
class StandoffReader:
    """
    Reader for the BRAT Standoff format.

    Parses annotation files (`.ann`) alongside text files (`.txt`) following
    the BRAT standoff annotation convention. Attempts to automatically resolve cases
    where annotation indices don't match the source text exactly

    Attributes:
        arguments: Configuration for the BRAT Standoff format reader.
    """

    arguments: StandoffArguments

    def _read_lines(self, text_source: FileSource, ann_file: FileSource) -> NERDocument | None:
        """
        Parse text and annotation files to create a NERDocument.

        Any empty documents with empty annotations are automatically filtered.

        Args:
            text_source: FileSource containing the text content.
            ann_file: FileSource containing the standoff annotations.

        Returns:
            `NERDocument`s parsed from the files or `None` if the text
            and annotation files are empty.

        Raises:
            ValueError: If an empty document with a non-empty annotation
                file is encountered
            ValueError: If an annotation offset mismatch occurs with
                multiple spans or an offset mismatch can't be
                automatically resolved.
        """

        text = text_source.read()

        if not text or text.isspace():
            annotations = ann_file.read()
            if annotations and not annotations.isspace():
                raise ValueError(
                    f"Found empty text document with annotations {{text: {text_source.path}, annotations: {ann_file.path}}}"
                )

            logger.warning(f"Skipping empty text document with no annotations: {text_source.path!r}")
            return None

        annotations = []
        for line in ann_file.lines():
            # Only process entity annotations
            if not line.startswith("T"):
                continue

            _, type_offset, expected_span = line.strip().split("\t", 2)
            trimmed_span = expected_span.strip()
            if trimmed_span != expected_span:
                logger.warning(f"Trimmed whitespace from span {expected_span!r} -> {trimmed_span!r}")
                expected_span = trimmed_span

            entity_type, offsets = type_offset.split(" ", 1)

            spans: list[Span] = []
            actual_spans = []
            for offset in offsets.split(";"):
                start, end = map(int, offset.split(" "))

                if self.arguments.offsets_without_newlines:
                    prefix = text[:start]
                    offset = prefix.count("\n")
                    start -= offset
                    end -= offset

                actual_spans.append(text[start:end])
                spans.append(Span(start, end))

            actual_span = " ".join(actual_spans)
            if expected_span != actual_span:
                if len(spans) > 1:
                    raise ValueError(
                        f"Annotation offset mismatch. Expected {expected_span!r} but got {actual_span!r} at offsets {spans}"
                    )

                # Fixes cases where annotations are completely mismatched
                if len(expected_span) != len(actual_span):
                    try:
                        # Try correcting additional whitespace in the span first
                        stripped = actual_span.strip()
                        start_correction = actual_span.index(stripped)
                    except ValueError:
                        # Otherwise, find the expected span within the text span if it is too long
                        start_correction = actual_span.index(expected_span)
                        stripped = expected_span

                    [original_span] = spans
                    spans[0] = Span(
                        original_span.start + start_correction,
                        original_span.stop - (len(actual_span) - len(stripped) - start_correction),
                    )

                    fixed_span = spans[0]
                    logger.warning(
                        f"Automatically fixed misaligned span {actual_span!r}: {original_span} -> {fixed_span} ({text[fixed_span.start : fixed_span.stop]})"
                    )
                else:
                    # Fixes offsets where labels are only mismatched due to an error in the starting offset
                    spans[0] = _fix_mismatched_offset(text, expected_span, actual_span, spans[0])

            annotations.append(Annotation(entity_type, tuple(spans)))

        # Standoff files generally contain full passages without explicit sentence segmentation
        return NERDocument([LabeledText(text, default_tagset(annotations), "passage")])

    def read_split(self, split: SplitData) -> Iterable[NERDocument]:
        """
        Reads a split and yields NERDocuments for each text-annotation pair.

        Args:
            split: Split instance with dictionary sources containing
                text and annotation files.

        Yields:
            NERDocument instances from reading each text-annotation pair.

        Raises:
            ValueError: If shard type does not match the expected format.
        """

        for shard in split.data:
            if not isinstance(shard, dict):
                raise TypeError(
                    f"{self.__class__.__name__} only supports dictionary sources, but found shard of type {type(shard)}"
                )

            # Ensure text files and ann files are aligned
            text_sources = shard["text"]
            ann_sources = {ann.path.stem: ann for ann in shard["annotations"]}

            for text_source in text_sources:
                document = self._read_lines(text_source, ann_sources[text_source.path.stem])
                if document is not None:
                    yield document


@ReaderRegistry.register("sofc_standoff")
@dataclass(slots=True)
class SofcStandoffReader:
    """
    Reader for the custom standoff format used by SOFC-Exp.

    Attributes:
        tagsets: List of NamedTagSet instances for entity and slot tags.
        arguments: Configuration for the standoff format reader.
    """

    _IGNORED_LINKS: ClassVar[set[str]] = {"experiment_variation", "same_experiment", "coreference"}

    arguments: SofcStandoffArguments

    tagsets: ClassVar[list[NamedTagSet]] = [NamedTagSet("entities"), NamedTagSet("slots")]

    def _read_sentences(self, text: FileSource, sentences: FileSource) -> list[tuple[str, str]]:
        """
        Reads sentence offsets and extracts sentences from full text.

        Args:
            text: FileSource containing the full document text.
            sentences: FileSource containing tab-separated sentence
                offsets.

        Returns:
            List of (sentence text, space after) tuples.
        """

        full_text = text.read()

        extracted_sentences: list[tuple[str, str]] = []
        previous_end = 0
        previous_text = None
        for _, _, start, end in csv.reader(sentences.lines(), delimiter="\t"):
            start = int(start)
            if previous_text is not None:
                extracted_sentences.append((previous_text, full_text[previous_end:start]))

            previous_end = int(end)
            previous_text = full_text[start:previous_end]

        assert previous_text is not None
        extracted_sentences.append((previous_text, ""))
        return extracted_sentences

    def _read_experiment_document(self, text: FileSource, sentences: FileSource, entities: FileSource) -> NERDocument:
        """
        Reads documents with entity and slot annotations.

        Args:
            text: FileSource containing the full document text.
            sentences: FileSource containing sentence offsets.
            entities: FileSource containing entity and slot annotations
                in BIO format.

        Returns:
            NERDocument with both entity and slot annotations across
            sentences.
        """

        extracted_sentences = self._read_sentences(text, sentences)

        tokenized_sentences = []
        annotated_sentences = []

        current_tokens = []
        current_token_spans = []
        current_entities = []
        current_slots = []

        last_sentence_id = None
        for sentence_id, _, start, end, bio_type, bio_slot in csv.reader(entities.lines(), delimiter="\t"):
            sentence_id = int(sentence_id)
            if last_sentence_id is None:
                last_sentence_id = sentence_id

            if sentence_id != last_sentence_id:
                tokenized_sentences.append(
                    LabeledTokens(current_tokens, {"entities": current_entities, "slots": current_slots})
                )
                sentence, space_after = extracted_sentences[last_sentence_id - 1]
                annotated_sentences.append(
                    LabeledText(
                        sentence,
                        {
                            "entities": bio_to_spans(current_entities, current_token_spans),
                            "slots": bio_to_spans(current_slots, current_token_spans),
                        },
                        space_after=space_after,
                    )
                )
                current_tokens = []
                current_token_spans = []
                current_entities = []
                current_slots = []
                last_sentence_id = sentence_id

            slot_tag = BIO.from_string(bio_slot)
            # NOTE: Removing this specific type since it occurs only once in the entire dataset
            # within a "entity_types_and_slots" file and not in the corresponding "frames" file.
            # Furthermore, the type is not mentioned in the paper
            if slot_tag.entity_type == "interconnect_material":
                logger.warning(f'Removed inconsistent "{bio_slot}" tag')
                slot_tag = BIO("O")

            current_entities.append(BIO.from_string(bio_type))
            current_slots.append(slot_tag)
            span = Span(int(start), int(end))
            current_token_spans.append(span)
            sentence, _ = extracted_sentences[last_sentence_id - 1]
            current_tokens.append(sentence[span.start : span.stop])

        if current_token_spans:
            tokenized_sentences.append(
                LabeledTokens(current_tokens, {"entities": current_entities, "slots": current_slots})
            )
            sentence, space_after = extracted_sentences[(last_sentence_id or 1) - 1]
            annotated_sentences.append(
                LabeledText(
                    sentence,
                    {
                        "entities": bio_to_spans(current_entities, current_token_spans),
                        "slots": bio_to_spans(current_slots, current_token_spans),
                    },
                    space_after=space_after,
                )
            )

        return NERDocument(annotated_sentences, tokenized_sentences)

    def _read_frame_document(self, text: FileSource, sentences: FileSource, frames: FileSource) -> NERDocument:
        """
        Reads frame annotation documents with experiment and slot annotations.

        Args:
            text: FileSource containing the full document text.
            sentences: FileSource containing sentence offsets.
            frames: FileSource containing experiment and frame
                annotations.

        Returns:
            The parsed NERDocument.
        """

        extracted_sentences = self._read_sentences(text, sentences)

        annotated_sentences: list[LabeledText] = []
        for sentence, space_after in extracted_sentences:
            annotated_sentences.append(LabeledText(sentence, {"entities": [], "slots": []}, space_after=space_after))

        lines = frames.lines()
        spans = []

        first_span_line = ""

        for line in csv.reader(lines, delimiter="\t"):
            # SPAN annotations are always at the start of each file
            if line[0] != "SPAN":
                first_span_line = "\t".join(line)
                break

            _, _, entity_type, sentence_id, start, end = line

            sentence_span = Span(int(start), int(end))
            sentence_id = int(sentence_id)
            spans.append((sentence_id, sentence_span))

            # Keep only the general entity type for consistency with other subset
            entity_type = entity_type.split(":")[0]

            # Add slot type from other subset that isn't explicitly defined in the frame annotation
            if entity_type == "EXPERIMENT":
                annotated_sentences[sentence_id - 1].labels["slots"].append(
                    Annotation("experiment_evoking_word", (sentence_span,))
                )

            annotated_sentences[sentence_id - 1].labels["entities"].append(Annotation(entity_type, (sentence_span,)))

        first_link_line = ""

        for line in csv.reader(itertools.chain((first_span_line,), lines), delimiter="\t"):
            # Add slot annotation layer but stop before link definitions
            if line[0] == "LINK":
                first_link_line = "\t".join(line)
                break

            if line[0] == "EXPERIMENT":
                continue

            _, slot_type, span_id = line
            sentence_id, sentence_span = spans[int(span_id) - 1]
            annotated_sentences[sentence_id - 1].labels["slots"].append(Annotation(slot_type, (sentence_span,)))

        for _, link_type, _, target in csv.reader(itertools.chain((first_link_line,), lines), delimiter="\t"):
            # Some link types don't correspond to slots
            if link_type in self._IGNORED_LINKS:
                continue

            sentence_id, sentence_span = spans[int(target) - 1]
            annotated_sentences[sentence_id - 1].labels["slots"].append(Annotation(link_type, (sentence_span,)))

        # NOTE: Slot annotations may not be in order but will be sorted by ConvertModule
        return NERDocument(annotated_sentences)

    def read_split(self, split: SplitData) -> Iterable[NERDocument]:
        """
        Reads a split and parses it into NERDocuments.

        Args:
            split: Split instance with dictionary sources containing
                text, sentences, and annotation files.

        Yields:
            The parsed NERDocument instances.

        Raises:
            ValueError: If the shard is not a dictionary.
            ValueError: If the number of files does not match across
                annotation layers.
        """

        for shard in split.data:
            if not isinstance(shard, dict):
                raise TypeError(
                    f"{self.__class__.__name__} only supports dictionary sources, but found shard of type {type(shard)}"
                )

            texts = shard["text"]
            text_source_count = len(texts)

            # Sanity check
            if not all(len(sources) == text_source_count for sources in shard.values()):
                raise ValueError("Different number of files encountered for different annotation layers")

            # Ensure text, sentence offset and annotation files are aligned
            sentences = {sentences.path.stem: sentences for sentences in shard["sentences"]}

            match self.arguments.label_source:
                case "entities":
                    entities = {entities.path.stem: entities for entities in shard["entities"]}
                    for text in texts:
                        text_name = text.path.stem
                        yield self._read_experiment_document(text, sentences[text_name], entities[text_name])

                case "frames":
                    frames = {frames.path.stem: frames for frames in shard["frames"]}
                    for text in texts:
                        text_name = text.path.stem
                        yield self._read_frame_document(text, sentences[text_name], frames[text_name])


def _get_annotation_type(source: FileSource) -> str:
    """
    Infers the annotation type from a file path.

    Args:
        source: FileSource pointing to an annotation file.

    Returns:
        Annotation type string ("broad" or "granular").
    """

    return base.parent.name if (base := source.path.parent.parent).name == "test" else base.name


@dataclass(slots=True)
class _AnnotatedSource:
    """
    Represents an annotated data source with its annotation type.

    Used internally by EBMNLPStandoffReader to track annotation sources
    for broad and granular annotation layers.

    Attributes:
        annotation_type: Type of annotations ("broad" or "granular").
        source: FileSource containing the annotation data.
    """

    annotation_type: Literal["broad", "granular"]
    source: FileSource


@ReaderRegistry.register("ebm_nlp_standoff")
@dataclass
class EBMNLPStandoffReader:
    """
    Reader for the EBM-NLP standoff format with broad and granular annotation layers.

    Attributes:
        tagsets: List of NamedTagSet instances for broad and granular
            tags.
        arguments: Configuration for the standoff format reader.
    """

    arguments: EBMNLPStandoffArguments

    tagsets: ClassVar[list[NamedTagSet]] = [NamedTagSet("broad"), NamedTagSet("granular")]

    def __post_init__(self) -> None:
        """Constructs label maps for broad and granular annotation layers."""
        self._broad_label_map = {
            name: {index: label for label, index in label_map.items()}
            for name, label_map in self.arguments.broad_label_map.items()
        }
        self._label_map = {
            name: {index: label for label, index in label_map.items()}
            for name, label_map in self.arguments.label_map.items()
        }

    def _read_document(
        self,
        text: FileSource,
        token_source: FileSource,
        annotations: dict[str, _AnnotatedSource],
    ) -> NERDocument:
        """
        Reads a complete EBM-NLP document with both annotation layers.

        Args:
            text: FileSource containing the original text.
            token_source: FileSource containing pre-tokenized text.
            annotations: Dictionary mapping annotation types to sources.

        Returns:
            NERDocument with both broad and granular annotations.

        Raises:
            ValueError: If an IOB label sequence length doesn't match
                the number of tokens.
        """

        original_text = text.read().strip()
        tokens = list(map(str.strip, token_source.lines()))
        token_spans = align_tokens_with_text(tokens, original_text)
        token_count = len(token_spans)

        broad_span_annotations = []
        span_annotations = []
        for name, source in annotations.items():
            if source.annotation_type == "broad":
                label_map = self._broad_label_map[name]
            else:
                label_map = self._label_map[name]

            bio_labels = [BIO.from_string(label_map[int(label_id)]) for label_id in source.source.lines()]
            if len(bio_labels) != token_count:
                raise ValueError(
                    f"IOB Label sequence must match the number of tokens in the document. Expected: {token_count}, actual: {len(bio_labels)}"
                )

            spans = bio_to_spans(bio_labels, token_spans)

            if source.annotation_type == "broad":
                broad_span_annotations.extend(spans)
            else:
                span_annotations.extend(spans)

        return NERDocument(
            [LabeledText(original_text, {"broad": broad_span_annotations, "granular": span_annotations}, "passage")]
        )

    def read_split(self, split: SplitData) -> Iterable[NERDocument]:
        """
        Reads a split and yields NERDocuments for the EBM-NLP dataset with broad and granular annotation layers.

        Args:
            split: Split instance with dictionary sources containing
                text, token, and annotation files.

        Yields:
            NERDocument instances from reading documents with both annotation layers.

        Raises:
            ValueError: If the shard is not a dictionary.
        """

        for shard in split.data:
            if not isinstance(shard, dict):
                raise TypeError(
                    f"{self.__class__.__name__} only supports dictionary sources, but found shard of type {type(shard)}"
                )

            # NOTE: EBM-NLP 2.0 does not include annotation files for some documents where all annotations are "O"
            # This requires constructing a sparse mapping between annotations and documents to ensure
            # that all are assigned to the correct split
            annotations = defaultdict(dict)
            for annotation in shard["broad_annotations"]:
                annotations[annotation.path.stem.removesuffix(".AGGREGATED")][_get_annotation_type(annotation)] = (
                    _AnnotatedSource("broad", annotation)
                )

            for annotation in shard["annotations"]:
                annotations[annotation.path.stem.removesuffix(".AGGREGATED")][_get_annotation_type(annotation)] = (
                    _AnnotatedSource("granular", annotation)
                )

            text_map = {source.path.stem: source for source in shard["text"]}
            tokens_map = {source.path.stem: source for source in shard["tokens"]}

            for document_id, annotation_sources in annotations.items():
                yield self._read_document(
                    text_map[document_id],
                    tokens_map[document_id],
                    annotation_sources,
                )


def _pubtator_text_line(line: str) -> tuple[int, Literal["a", "t"], str]:
    """
    Parses a PubTator text line.

    Args:
        line: PubTator text line in format "doc_id|type|text".

    Returns:
        Tuple of (document id, text type ("a" or "t"), text).

    Raises:
        ValueError: When a PubTator line is annotated with an unknown
            text type.
    """

    document_id, text_type, text = line.split("|", 2)
    if text_type not in ("a", "t"):
        raise ValueError(f"Unknown pubtator text type {text_type!r}")

    return int(document_id), text_type, text


@dataclass(slots=True)
class _PubtatorLine:
    """
    Represents a PubTator annotation line.

    Attributes:
        doc_id: Document identifier.
        start: Start offset of the annotation.
        end: End offset of the annotation.
        span: Text span of the annotation.
        label: Annotation type label.
        mesh_id: MeSH identifier of the entity.
    """

    doc_id: int
    start: int
    end: int
    span: str
    label: str
    # MeSH ID
    mesh_id: str


@ReaderRegistry.register("pubtator")
@dataclass(slots=True)
class PubtatorReader(LineReader):
    """Reader for the PubTator format with document-level title and abstract annotations."""

    def _read_lines(self, lines: FileSource) -> Iterable[NERDocument]:
        """
        Reads PubTator format and yields NERDocuments with title and abstract passages.

        Args:
            lines: FileSource containing PubTator format text.

        Yields:
            NERDocument instances with title and abstract passages and their annotations.

        Raises:
            ValueError: If an annotation offset mismatch cannot be
                automatically resolved.
            ValueError: When a PubTator line is annotated with an
                unknown text type.
        """

        parser = RowParser(_PubtatorLine)

        document_ids = set()
        for segment in space_separated_segments(lines.lines()):
            document_id, _, first_passage = _pubtator_text_line(segment[0])
            if document_id in document_ids:
                logger.warning(f"Skipping duplicate document ID {document_id} in file")
                continue

            document_ids.add(document_id)

            *_, second_passage = _pubtator_text_line(segment[1])
            combined_text = first_passage + "\n" + second_passage

            # Add one to account for newline between the first and second passage
            first_offset = len(first_passage) + 1

            first_labels = []
            second_labels = []
            for line in segment[2:]:
                span = parser.validate_row(line.split("\t"))
                text_span = combined_text[span.start : span.end]
                if text_span != span.span:
                    if len(text_span) != len(span.span):
                        raise ValueError(
                            f"Annotation offset mismatch. Expected {span.span!r} but got {text_span!r} at offset ({span.start}, {span.end})"
                        )

                    logger.warning(f"Annotation inconsistency: Expected {span.span!r}, found {text_span!r}")

                if span.start < first_offset:
                    first_labels.append(Annotation(span.label, (Span(span.start, span.end),)))
                else:
                    second_labels.append(
                        Annotation(span.label, (Span(span.start - first_offset, span.end - first_offset),))
                    )

            # Add a newline between the title and the abstract when used as a single document
            yield NERDocument(
                [
                    LabeledText(first_passage, default_tagset(first_labels), "passage", space_after="\n"),
                    LabeledText(second_passage, default_tagset(second_labels), "passage"),
                ]
            )


@dataclass(slots=True)
class DatasetReader:
    """
    Reader for Huggingface Dataset sources with NER annotations.

    Reads datasets in various formats and converts them to NERDocuments,
    handling detokenization, BIO tags, and label mapping.

    Attributes:
        config: Dataset conversion configuration.
        tagsets: Optional list of NamedTagSet instances for tag
            configuration.
    """

    config: DatasetConvertStep
    tagsets: list[NamedTagSet] | None = None

    def __post_init__(self) -> None:
        """Initializes tagsets from label maps if provided."""
        self.tagsets = (
            None
            if self.config.tagsets is None
            else [
                # Strings are just column names
                NamedTagSet(name, tagset)
                if isinstance(tagset, str)
                # Invert label map for index to BIO conversion
                else NamedTagSet(
                    name,
                    tagset.column,
                    None if tagset.label_map is None else {index: label for label, index in tagset.label_map.items()},
                )
                for name, tagset in self.config.tagsets.items()
            ]
        )

    def read_split(self, split: SplitData) -> Iterable[NERDocument]:
        """
        Reads a split and yields NERDocuments from a Huggingface dataset.

        Args:
            split: Split instance with DatasetSource containing the
                dataset.

        Yields:
            NERDocument instances from detokenizing and processing dataset rows.

        Raises:
            ValueError: If an input shard is not a HuggingFace dataset.
        """

        detokenizer = Detokenizer(split.language, self.config.detokenizer_type)
        sequence_type = self.config.sequence_type

        for shard in split.data:
            if not isinstance(shard, DatasetSource):
                raise TypeError(f"DatasetReader can only process datasets, but found non-dataset shard: {shard}")

            dataset = shard.dataset

            tagsets = [NamedTagSet()] if self.tagsets is None else self.tagsets

            for row in dataset.select_columns(
                [shard.text_column, *{tagset.column or shard.default_tag_column for tagset in tagsets}]
            ).to_iterable_dataset():
                mapped_tagsets = {}
                for tagset in tagsets:
                    tag_column = tagset.column or shard.default_tag_column
                    label_map = tagset.label_map
                    tags = row[tag_column]
                    if label_map is None:
                        if self.config.bio_type == "iob_type_only":
                            # BIO tag format with an implicit "I" position. Used by Polyglot-NER
                            bio_tags = [BIO("O") if iob == "O" else BIO("I", iob) for iob in tags]
                        else:
                            bio_tags = list(map(BIO.from_string, tags))
                        mapped_tagsets[tagset.name] = bio_tags
                    else:
                        mapped_tagsets[tagset.name] = [label_map[tag] for tag in tags]

                # NOTE: Assumes all text columns are sentence level
                yield detokenizer.tokens_to_document(
                    [LabeledTokens(row[shard.text_column], mapped_tagsets, sequence_type)]
                )


@ReaderRegistry.register("dataset_spans")
@dataclass(slots=True)
class DatasetSpanReader:
    """
    Reader for Huggingface Dataset sources with pre-aligned span annotations.

    Reads datasets where spans are already aligned with text content.
    """

    def read_split(self, split: SplitData) -> Iterable[NERDocument]:
        """
        Reads a split and yields NERDocuments from a Huggingface dataset with pre-aligned spans.

        Args:
            split: Split instance with DatasetSource instances.

        Yields:
            NERDocument instances with text and aligned span annotations.

        Raises:
            ValueError: If an input shard is not a dataset.
            ValueError: If span mismatch is detected.
        """

        for shard in split.data:
            if not isinstance(shard, DatasetSource):
                raise TypeError(f"DatasetSpanReader can only process datasets, but found non-dataset shard: {shard}")

            dataset = shard.dataset

            for row in dataset.select_columns([shard.text_column, shard.default_tag_column]).to_iterable_dataset():
                annotations = []
                text = row[shard.text_column]
                for annotation in row[shard.default_tag_column]:
                    expected = annotation["name"]
                    span = Span(*annotation["span"])
                    actual = text[span.start : span.stop]

                    if actual != expected:
                        raise ValueError(f"Span mismatch {actual!r} : {expected!r}")

                    annotations.append(Annotation(annotation["type"], (span,)))

                yield NERDocument([LabeledText(text, default_tagset(annotations))])


def reader_from_config(reader_config: ReaderConfiguration) -> NERReader:
    """
    Initializes a registered reader from its configuration.

    Args:
        reader_config: ReaderConfiguration specifying the reader type and parameters.

    Returns:
        A configured reader instance for the specified format.
    """

    # Manual collection of arguments instead to avoid recursive behavior of dataclasses.asdict
    parameters = {field.name: getattr(reader_config, field.name) for field in dataclasses.fields(reader_config)}
    reader_type = parameters.pop("type")
    return ReaderRegistry.get(reader_type)(**parameters)
