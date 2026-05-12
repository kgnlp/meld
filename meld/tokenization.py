"""Text detokenization, token alignment and sentence splitting."""

import dataclasses
import logging
import uuid
from importlib.resources.abc import Traversable
from pathlib import Path
from types import TracebackType
from typing import Self

import pyarrow as pa
from pyarrow import RecordBatch, parquet
from pyarrow.parquet import ParquetWriter

from meld.document import BIO, Annotation, LabeledText, LabeledTokens, NERDocument, Span
from meld.manifest import DetokenizerType, SentenceBoundaryType

try:
    from wtpsplit import SaT  # noqa: I001 pyright: ignore

    # Only imported if SaT is available
    import torch
except ImportError:
    SaT = None
    torch = None


logger = logging.getLogger("meld")

SENTENCE_SPLIT_SCHEMA = pa.schema(
    [
        pa.field("sequence_id", pa.binary()),
        pa.field(
            "sentence_offsets",
            pa.list_(
                pa.struct(
                    [
                        pa.field("start", pa.uint32(), nullable=False),
                        pa.field("stop", pa.uint32(), nullable=False),
                    ]
                )
            ),
        ),
    ]
)


def _find_next(sentence_span: Span, current_labels: list[Annotation], previous_start: int) -> int | None:
    """
    Find the next label index that starts after the given sentence span.

    Args:
        sentence_span: The span of the current sentence being processed.
        current_labels: The list of annotations for the current text
            segment.
        previous_start: The starting index to search from.

    Returns:
        The index of the next label, or None if a label spans across the
        sentence boundary.
    """

    index = len(current_labels)
    for index in range(previous_start, index):
        label = current_labels[index]
        end_position = label.spans[-1].stop
        if label.spans[0].start < sentence_span.stop and end_position > sentence_span.stop:
            return None
        elif label.spans[0].start > sentence_span.stop:
            return index

    return index + 1


def _find_next_or_merge(
    sentence_span: Span, labels: dict[str, list[Annotation]], previous_starts: dict[str, int]
) -> dict[str, int] | None:
    """
    Find the next label indices for all tagsets or merge spans if they cross sentence boundaries.

    Args:
        sentence_span: The span of the current sentence being processed.
        labels: Annotations of each tagset.
        previous_starts: Mapping of tagset names to their current
            starting indices in the document.

    Returns:
        Dictionary of new starting indices for each tagset, or None if a
        label spans across the sentence boundary.
    """

    new_starts = {}
    for label_set, current_labels in labels.items():
        next_index = _find_next(sentence_span, current_labels, previous_starts[label_set])
        if next_index is None:
            return None

        new_starts[label_set] = next_index

    return new_starts


def _fit_spans(spans: tuple[Span, ...], offset: int) -> tuple[Span, ...]:
    """
    Adjust spans by subtracting the given offset from their start and stop positions.

    Args:
        spans: Spans to adjust.
        offset: The offset to subtract from each span's start and stop
            positions.

    Returns:
        Adjusted spans.
    """

    return tuple(Span(span.start - offset, span.stop - offset) for span in spans)


def _align_tokens_to_spans(text: str, spans: list[Span], labeled_tokens: LabeledTokens) -> list[LabeledTokens]:
    """
    Align tokenized tokens to sentence spans while ensuring tokens don't cross sentence boundaries.

    Args:
        text: The original text containing the sentence spans.
        spans: List of sentence spans to align tokens to.
        labeled_tokens: The labeled tokens to segment.

    Returns:
        List of labeled tokens segmented by sentence spans.

    Raises:
        ValueError: If a token spans beyond a sentence boundary or
            tokens cannot be found in text.
    """

    tokens = labeled_tokens.tokens
    if not spans:
        if tokens:
            raise ValueError(f"Tokens found but spans were empty: {tokens}")
        return [labeled_tokens]

    token_offsets = []
    start = 0
    for token in tokens:
        start = text.find(token, start)
        if start == -1:
            raise ValueError(f"Could not find {token} in suffix {text[start:]!r} of text {text!r}")

        stop = start + len(token)
        token_offsets.append(Span(start, stop))
        start = stop

    segmented_tokens = []
    token_index = 0
    for segment_span in spans:
        start_index = token_index
        for token_index in range(token_index, len(tokens)):
            span = token_offsets[token_index]
            if span.start >= segment_span.stop:
                break
            elif span.stop > segment_span.stop:
                raise ValueError(
                    f"Token {tokens[token_index]!r} spans beyond the end of sentence {text[segment_span.start : segment_span.stop]!r}"
                )
        else:
            token_index += 1

        segmented_tokens.append(
            LabeledTokens(
                tokens[start_index:token_index],
                {tagset: labels[start_index:token_index] for tagset, labels in labeled_tokens.labels.items()},
                labeled_tokens.sequence_type,
            )
        )

    return segmented_tokens


def _split_document(document: NERDocument, sentence_offsets: list[list[Span]]) -> tuple[NERDocument, list[list[Span]]]:
    """
    Split a document's spans into sentences based on the provided sentence offsets.

    Args:
        document: The document to split.
        sentence_offsets: List of sentence offset lists for each
            document segment.

    Returns:
        The modified document and adjusted offsets.
    """

    adjusted_offsets = []
    segmented_spans = []

    for segment, offsets in zip(document.spans, sentence_offsets):
        current_sentence = 0
        previous_starts = {tagset: 0 for tagset in segment.labels}
        while current_sentence < len(offsets):
            current_sentence_span = offsets[current_sentence]
            new_label_starts = _find_next_or_merge(current_sentence_span, segment.labels, previous_starts)
            if new_label_starts is None:
                offsets = [
                    *offsets[:current_sentence],
                    Span(current_sentence_span.start, offsets[current_sentence + 1].stop),
                    *offsets[current_sentence + 2 :],
                ]
                continue

            segmented_spans.append(
                LabeledText(
                    segment.text[current_sentence_span.start : current_sentence_span.stop],
                    {
                        tagset: [
                            Annotation(annotation.label, _fit_spans(annotation.spans, current_sentence_span.start))
                            for annotation in labels[previous_starts[tagset] : new_label_starts[tagset]]
                        ]
                        for tagset, labels in segment.labels.items()
                    },
                    space_after=segment.text[current_sentence_span.stop : offsets[current_sentence + 1].start]
                    if current_sentence < len(offsets) - 1
                    else "",
                )
            )
            previous_starts = new_label_starts
            current_sentence += 1

        adjusted_offsets.append(offsets)

    if document.bio is not None:
        # Align pre-tokenized text if available
        document.bio = [
            segmented_tokens
            for labeled_span, labeled_tokens, spans in zip(document.spans, document.bio, adjusted_offsets)
            for segmented_tokens in _align_tokens_to_spans(labeled_span.text, spans, labeled_tokens)
        ]

    document.spans = segmented_spans

    return document, adjusted_offsets


# Random UUID v4 namespace for generated sequence_ids
_SENTENCE_SPLIT_UUID_NAMESPACE = uuid.UUID("c9d3a311-7c27-4a86-b6fb-a52aa9f53b63")


class SentenceSplitter:
    """
    Sentence tokenizer that splits documents into sentences using either a Segment Any Text (SAT) [1] model
    or pre-computed sentence boundaries from Parquet files.

    Requires the optional `wtpsplit` dependency to be installed unless `read_spans` is `True` (e.g. by installing with the "sentence-segmentation" extra enabled).

    # References

    [1] Markus Frohmann, Igor Sterner, Ivan Vulić, Benjamin Minixhofer, and Markus Schedl. 2024. [Segment Any Text: A Universal Approach for Robust, Efficient and Adaptable Sentence Segmentation](https://aclanthology.org/2024.emnlp-main.665/). In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 11908–11941, Miami, Florida, USA, November. Association for Computational Linguistics.

    Args:
        sentence_boundaries: Type of sentence boundaries.
        sentence_span_file: Optional path to a Parquet file with
            pre-computed sentence spans used if `read_spans` is
            `True`.
        sat_model: Model name for `wtpsplit` for segmenting text if
            `read_spans` is `False`.
        read_spans: Whether to read from pre-computed span file
            instead of splitting.

    Raises:
        ValueError: When attempting to tokenize when full sentence
            boundaries already exist (`sentence_boundaries` is set
            to "full").
        ValueError: When `read_spans` is set to `True` but no
            `sentence_span_file` is provided.
        ImportError: If the optional `wtpsplit` dependency is not
            installed.
    """

    _BUFFER_SIZE = 256

    def __init__(
        self,
        sentence_boundaries: SentenceBoundaryType,
        sentence_span_file: Path | Traversable | None = None,
        sat_model: str = "sat-12l-sm",
        read_spans: bool = True,
    ) -> None:
        if sentence_boundaries == "full":
            raise ValueError("Sentence tokenization only supports splits that are not already segmented into sentences")

        self._split_prefix = None if sentence_span_file is None else sentence_span_file.name.encode()
        self._sentence_span_path = sentence_span_file
        self._sentence_span_file = None
        self._read_spans = read_spans

        self._boundaries = None
        self._writer = None

        self._sequence_index = 0

        if read_spans:
            if self._sentence_span_path is None:
                raise ValueError("`sentence_span_file` must not be `None` if `read_spans` is `True`")

            self._sat = None
            return

        if SaT is None or torch is None:
            raise ImportError('Sentence splitting requires optional dependency "wtpsplit" to be installed')

        self._sat = SaT(sat_model)
        if torch.cuda.is_available():
            self._sat.to("cuda")

        self._parquet_batch = {
            "sequence_id": [],
            "sentence_offsets": [],
        }

    def __enter__(self) -> Self:
        if self._sentence_span_path is not None:
            if self._read_spans:
                self._sentence_span_file = self._sentence_span_path.open("rb")
                self._boundaries = parquet.read_table(self._sentence_span_file)
            else:
                if not isinstance(self._sentence_span_path, Path):
                    raise ValueError("sentence_span_path must be a Path object for writing")
                self._sentence_span_file = self._sentence_span_path.open("wb")
                self._writer = ParquetWriter(self._sentence_span_file, SENTENCE_SPLIT_SCHEMA, compression="zstd")

        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        # Write final batch if necessary
        if self._writer is not None and self._parquet_batch["sequence_id"]:
            self._writer.write_batch(RecordBatch.from_pydict(self._parquet_batch, SENTENCE_SPLIT_SCHEMA))
        if self._sentence_span_file is not None:
            if self._writer is not None:
                self._writer.close()
            self._sentence_span_file.close()

    def _current_uuid(self) -> bytes:
        """
        Generate a reproducible UUID v5 for the current sequence.

        Returns:
            The generated UUID as bytes.
        """

        if self._split_prefix is None:
            raise ValueError("A split prefix needs to be defined for gerating UUIDs for the sample")

        return uuid.uuid5(
            _SENTENCE_SPLIT_UUID_NAMESPACE, self._split_prefix + self._sequence_index.to_bytes(64, "little")
        ).bytes

    def sentence_tokenize(self, document: NERDocument) -> NERDocument:
        """
        Tokenize the document into sentences and update its spans accordingly.

        Args:
            document: The document to sentence-tokenize.

        Returns:
            The document with spans split into sentences.
        """

        if self._sat is None:
            assert self._boundaries is not None
            end_index = self._sequence_index + len(document.spans)
            offsets = self._boundaries.column("sentence_offsets")[self._sequence_index : end_index]
            document, _ = _split_document(
                document,
                [
                    [Span(*sentence_span.values()) for sentence_span in sentence_offsets.as_py()]
                    for sentence_offsets in offsets
                ],
            )
            self._sequence_index = end_index
            return document

        segment_ids = self._parquet_batch["sequence_id"]
        sentence_offsets = self._parquet_batch["sentence_offsets"]

        offsets = []

        segments = [annotated_text.text for annotated_text in document.spans]

        for sentences, segment in zip(
            self._sat.split(segments, strip_whitespace=True, split_on_input_newlines=False), segments
        ):
            assert segment is not None, "Text field must not be None"
            offsets.append(align_tokens_with_text(sentences, segment))
            self._sequence_index += 1

        document, offsets = _split_document(document, offsets)
        if self._writer is None:
            return document

        sentence_offsets.extend([list(map(dataclasses.asdict, spans)) for spans in offsets])
        segment_ids.extend([self._current_uuid()] * len(offsets))

        if len(sentence_offsets) > self._BUFFER_SIZE:
            if self._writer is not None:
                self._writer.write_batch(RecordBatch.from_pydict(self._parquet_batch, SENTENCE_SPLIT_SCHEMA))

            self._parquet_batch = {
                "sequence_id": [],
                "sentence_offsets": [],
            }

        return document


def align_tokens_with_text(tokens: list[str], text: str) -> list[Span]:
    """
    Align token spans with the original text.

    Args:
        tokens: List of tokens to align.
        text: The original text to align tokens against.

    Returns:
        List of spans of each token's position in the text.

    Raises:
        ValueError: If tokens cannot be aligned with the text.
    """

    spans = []
    position = 0
    for token in tokens:
        # Consume whitespace between tokens
        while text[position].isspace():
            position += 1

        if not text[position : position + len(token)] == token:
            raise ValueError(f"Failed to align tokens {tokens} with text {text!r}")
        end = position + len(token)
        spans.append(Span(position, end))
        position = end

    return spans


def bio_to_spans(bio_labels: list[BIO], token_spans: list[Span]) -> list[Annotation]:
    """
    Convert BIO-formatted labels to annotated spans based on their position in the untokenized document.

    Args:
        bio_labels: List of BIO tags.
        token_spans: List of token spans corresponding to the BIO
            labels.

    Returns:
        List of annotations with labels and their spans.

    Raises:
        ValueError: If a BIO tag with position "B" or "I" is encountered
            without an entity type.
    """

    annotations: list[tuple[str, Span]] = []
    current_tag: tuple[str, Span] | None = None
    previous_tag = BIO("O")
    for tag, span in zip(bio_labels, token_spans):
        if current_tag and (tag.position in {"O", "B"} or tag.entity_type != previous_tag.entity_type):
            annotations.append(current_tag)
            current_tag = None

        if tag.position in {"B", "I"}:
            if current_tag:
                # Advance the stop index
                label, current_span = current_tag
                current_tag = (label, Span(current_span.start, span.stop))
            elif tag.entity_type is None:
                raise ValueError(f"BIO tag without type encountered: {tag}")
            else:
                current_tag = (tag.entity_type, span)

        previous_tag = tag

    if current_tag:
        annotations.append(current_tag)

    return [Annotation(label, (span,)) for label, span in annotations]


# Note: zh-classical is non-standard and unique to WikiANN
_CHARACTER_LEVEL = {"cmn", "jpn", "zho", "yue", "zh-classical"}
_WIKIANN_HASHES = _CHARACTER_LEVEL


class Detokenizer:
    """
    Detokenizes tokenized text into a string with aligned entity spans.

    Handles language-specific detokenization rules including character-level languages
    (Chinese, Japanese, etc.) and various special modes like WikiANN hash replacement.
    Currently text in most language is detokenized by simply joining tokens by whitespace.

    Args:
        language: The language of the tokens for selecting the
            appropriate detokenization strategy.
        detokenizer_type: The type of detokenization to perform. Options:
            - `"whitespace"` to join by whitespace
            - `"concatenate"` to concatenate tokens without a delimiter
            - `"wikiann"` for WikiANN specific preprocessing
    """

    def __init__(self, language: str, detokenizer_type: DetokenizerType = "whitespace") -> None:
        self.language = language
        self.separator = "" if detokenizer_type == "concatenate" or language in _CHARACTER_LEVEL else " "
        self.detokenizer_type = detokenizer_type
        self._wikiann_hash_replacement = detokenizer_type == "wikiann" and language in _WIKIANN_HASHES

    def _whitespace_detokenize(self, tokens: list[str]) -> tuple[str, list[Span]]:
        """
        Detokenize tokens by simple joining with whitespaces.

        Args:
            tokens: The list of tokens to detokenize.

        Returns:
            A tuple of the detokenized text and corresponding spans of
            the tokens in the detokenized text.
        """

        delimiter_length = len(self.separator)
        text = self.separator.join(tokens)
        start = 0
        spans = []
        for length in map(len, tokens):
            spans.append(Span(start, start + length))
            # Length + delimiter
            start += length + delimiter_length

        return text, spans

    def _preprocess_tokens(self, tokens: list[str]) -> list[str]:
        """
        Preprocesses tokens before detokenization. Currently only handles WikiANN hash replacement, if enabled

        Args:
            tokens: List of tokens to preprocess.

        Returns:
            Preprocessed tokens.
        """

        # Replace hash tokens in WikiANN which should be spaces
        if self._wikiann_hash_replacement:
            return [" " if token == "#" else token for token in tokens]

        return tokens

    def detokenize(self, tokens: list[str]) -> tuple[str, list[Span]]:
        """
        Detokenize a list of tokens into text with aligned spans.

        Args:
            tokens: List of tokens to detokenize.

        Returns:
            The detokenized text and spans for each token within it.
        """

        tokens = self._preprocess_tokens(tokens)
        return self._whitespace_detokenize(tokens)

    def detokenize_bio(
        self, document: list[LabeledTokens], original_text: list[str] | None = None
    ) -> list[LabeledText]:
        """
        Detokenize a BIO-formatted document. If an `original_text` is given, the detokenizer will simply align tokens with the given source text.

        Args:
            document: The `LabeledTokens` to detokenize.
            original_text: Optional original text for alignment.

        Returns:
            A list of `LabeledText` objects with detokenized text and
            spans.

        Raises:
            ValueError: If tokens cannot be aligned and fallback to
                whitespace tokenization fails.
        """

        detokenized = []
        for i, sentence in enumerate(document):
            sentence.tokens = self._preprocess_tokens(sentence.tokens)

            if original_text is None:
                text, sentence_offsets = self._whitespace_detokenize(sentence.tokens)
            else:
                text = original_text[i]
                try:
                    sentence_offsets = align_tokens_with_text(sentence.tokens, text)
                except ValueError as error:
                    logger.warning(
                        f"Could not align text with tokens, falling back to whitespace tokenization: {error}"
                    )
                    text, sentence_offsets = self._whitespace_detokenize(sentence.tokens)

            detokenized.append(
                LabeledText(
                    text,
                    {tagset: bio_to_spans(labels, sentence_offsets) for tagset, labels in sentence.labels.items()},
                )
            )

        return detokenized

    def tokens_to_document(
        self, labeled_tokens: list[LabeledTokens], original_text: list[str] | None = None
    ) -> NERDocument:
        """
        Convert labeled tokens to a `NERDocument`.

        Args:
            labeled_tokens: The list of LabeledTokens to convert.
            original_text: Optional original text for alignment.

        Returns:
            A `NERDocument` containing both the labeled tokens and
            detokenized text with aligned entity spans.
        """

        return NERDocument(self.detokenize_bio(labeled_tokens, original_text), labeled_tokens)
