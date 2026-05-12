"""CoNLL format definitions and readers."""

import dataclasses
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Annotated, Any, ClassVar, Protocol, Self, runtime_checkable

from pydantic import BeforeValidator, PlainSerializer, PlainValidator, TypeAdapter

from meld.document import BIO, BIOField
from meld.registry import Registry


def maybe_empty(empty_value: str = "_") -> BeforeValidator:
    """
    Create a Pydantic BeforeValidator that converts strings containing a CoNLL-style empty value to `None`.

    The validator will raise a `ValueError` if the input string is the empty string unless `empty_value` is also set to the empty string

    Args:
        empty_value: The string value representing empty fields.

    Returns:
        A Pydantic BeforeValidator that transforms empty values to None.
    """

    def transform(string: str) -> str | None:
        if not string and empty_value:
            raise ValueError(f"Value must be non-empty. All empty values must contain {empty_value!r} explicitly")
        return string if string != empty_value else None

    return BeforeValidator(transform)


type MaybeUnderscore = Annotated[str | None, maybe_empty()]
type MaybeHyphen = Annotated[str | None, maybe_empty("-")]


def key_value_list(list_sep: str = "|", key_value_sep: str = "=", allow_empty: bool = True) -> BeforeValidator:
    """
    Create a Pydantic BeforeValidator for parsing key-value list strings into dictionaries.

    The input string is parsed as a list of key-value pairs, where each pair is separated
    by `list_sep` and the key and value within each pair are separated by `key_value_sep`.

    Args:
        list_sep: Separator between key-value pairs.
        key_value_sep: Separator between keys and values.
        allow_empty: Whether an empty list input should be allowed or
            raise a `ValueError`.

    Returns:
        A Pydantic BeforeValidator that parses key-value list strings.

    Raises:
        ValueError: If `list_sep` and `key_value_sep` are the same.
        ValueError: In the returned validator if the input is `None` and `allow_empty` is `False`.
    """

    if list_sep == key_value_sep:
        raise ValueError(f"List and key value separator need to differ but where {list_sep!r} == {key_value_sep!r}")

    def transform(string: str | None) -> dict[str, str] | None:
        if string is None:
            if not allow_empty:
                raise ValueError("Key-value list can not be empty")
            return None

        return dict(pair.split(key_value_sep, 1) for pair in string.split(list_sep))

    return BeforeValidator(transform)


@dataclass(slots=True)
class CoNLLMetadata:
    """
    Stores CoNLL format metadata, handling both regular comments and metadata encoded as key-value pairs.

    Attributes:
        meta: Dictionary of key-value metadata.
        comments: List of other comment strings.
    """

    meta: dict[str, str] = field(default_factory=dict)
    comments: list[str] = field(default_factory=list)

    def get_meta(self, metadata_key: str) -> str | None:
        """
        Retrieve metadata by key.

        Args:
            metadata_key: The key to look up.

        Returns:
            The metadata value, or None if not found.
        """

        return self.meta.get(metadata_key)

    def __getitem__(self, metadata_key: str) -> str:
        """
        Retrieve metadata by key.

        Args:
            metadata_key: The key to look up.

        Returns:
            The metadata value.

        Raises:
            KeyError: If the key is not found.
        """

        return self.meta[metadata_key]

    def __contains__(self, metadata_key: str) -> bool:
        """
        Check if a metadata key exists.

        Args:
            metadata_key: The key to check.

        Returns:
            `True` if the key exists in the key-value store.
        """

        return metadata_key in self.meta

    @classmethod
    def with_key_value(cls, comments: Iterable[tuple[str, str] | str]) -> Self:
        """
        Create CoNLLMetadata from an iterable of plain comments and parsed key-value pairs.

        Args:
            comments: Iterable of tuples (key, value) or plain comment
                strings.

        Returns:
            A new CoNLLMetadata instance.
        """

        meta = {}
        general = []
        for comment in comments:
            if isinstance(comment, tuple):
                meta[comment[0]] = comment[1]
            else:
                general.append(comment)

        return cls(meta, general)


@runtime_checkable
class CommentedCoNLL(Protocol):
    """
    Protocol for CoNLL-style formats that support comment-based metadata."""

    @staticmethod
    def parse_comments(comments: list[str]) -> CoNLLMetadata:
        """
        Parse CoNLL metadata from comments.

        Args:
            comments: List of comment strings.

        Returns:
            Parsed CoNLLMetadata.
        """
        ...

    @staticmethod
    def is_document_start(metadata: CoNLLMetadata) -> bool:
        """
        Check if metadata indicates the start of a new document.

        Args:
            metadata: The CoNLLMetadata to check.

        Returns:
            True at the start of a new document.
        """
        ...


@runtime_checkable
class ColumnarCoNLL(Protocol):
    """
    Protocol for CoNLL-style formats that support column-based segmentation."""

    @staticmethod
    def segment_columns(rows: Iterable[dict[str, Any]]) -> Iterable[list[dict[str, Any]]]:
        """
        Segment sentence indexed rows into sentence-level groups.

        Args:
            rows: Iterable of indexed rows.

        Returns:
            Iterable of sentence-level row groups.
        """
        ...


@runtime_checkable
class DocumentSeparatedCoNLL(Protocol):
    """
    Protocol for CoNLL-style formats separated by document markers.

    Attributes:
        docstart: Class variable indicating the document start marker.
        ignore_docstart: Class variable indicating whether to ignore
            document start markers.
    """

    docstart: ClassVar[str]
    ignore_docstart: ClassVar[bool]


type CommonRowType = object | CommentedCoNLL | DocumentSeparatedCoNLL


class CoNLLDialectRegistry[T: CommonRowType](Registry[T]):
    """Registry for CoNLL dialects."""


class CoNLLColumnarRegistry[T: ColumnarCoNLL](Registry[T]):
    """Registry for CoNLL dialects."""


@CoNLLDialectRegistry.register("conll")
@dataclass(slots=True)
class CoNLL:
    """
    Simple CoNLL-style format with form and NER tag columns.

    Attributes:
        form: The token text.
        ner: The NER tag in BIO format.
    """

    form: str
    ner: BIOField


@CoNLLDialectRegistry.register("conll_bio_first")
@dataclass(slots=True)
class CoNLLBioFirst:
    """
    Simple CoNLL-style format in which NER tags are stored in the first column.

    Attributes:
        ner: The NER tag in BIO format.
        form: The token text.
    """

    ner: BIOField
    form: str


@CoNLLDialectRegistry.register("conll_with_pos")
@dataclass(slots=True)
class CoNLLWithPOS:
    """
    CoNLL-style format with POS and NER tags.

    Attributes:
        form: The token text.
        pos: The part-of-speech tag.
        ner: The NER tag in BIO format.
    """

    form: str
    pos: str
    ner: BIOField


def _strip_word_position(character_with_word_position: str) -> str:
    """
    Strip the word position index from a character token.

    The WeiboNER format annotates characters with indices indicating their position within a word,
    which this function removes.

    Args:
        character_with_word_position: Character token followed by an
            optional position index.

    Returns:
        The character without the position index.

    Raises:
        ValueError: If the token format is unexpected.
    """

    # Most WeiboNER lines start with a single character followed by digits indicating their index within a word
    if character_with_word_position[1:].isnumeric():
        return character_with_word_position[0]

    # Some lines start with two initial replacement characters instead and are extracted via a simple regex
    if character_with_word_position.startswith("�"):
        replacement_match = re.fullmatch(r"(�+)\d+", character_with_word_position)

        if replacement_match is not None:
            return replacement_match.group(1)

    raise ValueError(
        f"Expected a single character followed by one or multiple digits but found {character_with_word_position!r}"
    )


@CoNLLDialectRegistry.register("conll_weibo_ner")
@dataclass(slots=True)
class CoNLLWeiboNER:
    """
    CoNLL-style format from the WeiboNER dataset in which word position indices are stored as part of the word form column.
    Word position indices are removed during deserialization.

    Attributes:
        form: The token form/word with word position indices removed.
        ner: The NER tag in BIO format.
    """

    form: Annotated[str, BeforeValidator(_strip_word_position)]
    ner: BIOField


def _herodotos_inverted_bio_from_string(string: str) -> BIO:
    """
    Parse inverted BIO format string (e.g., "PERS-B" instead of "B-PERS"). This function additionally resolves Herodotos-Project-NER using "0" instead of "O" for outside tags

    Args:
        string: The inverted BIO string to parse.

    Returns:
        A parsed BIO instance.
    """

    if string == "0":
        return BIO("O")

    return BIO(*string.split("-")[::-1])


@CoNLLDialectRegistry.register("conll_herodotos")
@dataclass(slots=True)
class CoNLLHerodotos:
    """
    Two-column CoNLL-style format used by the Herodotos-Project-NER dataset. It is similar to `CoNLLBioFirst` but with inverted IOB tags (e.g., PERS-B instead of B-PERS) and "0" instead of "O".

    Attributes:
        ner: The Named Entity Recognition tag in inverted BIO format.
        form: The token form/word.
    """

    ner: Annotated[BIO, BeforeValidator(_herodotos_inverted_bio_from_string), PlainSerializer(str)]
    form: str


def conllu_id(token_id: str | float | Decimal) -> int | Decimal | range:
    """
    Parses CoNLL-U token IDs.

    Args:
        token_id: Token ID string representation (may contain hyphen for
            multi-word tokens or decimal for empty nodes).

    Returns:
        An integer for simple IDs, Decimal for empty nodes, or range for multi-
        word tokens.

    Raises:
        ValueError: When a decimal ID is less than or equal to 0.
    """

    if isinstance(token_id, str):
        if len(id_range := token_id.split("-", 1)) > 1:
            return range(*map(int, id_range))
        try:
            return int(token_id)
        except ValueError:
            decimal = Decimal(token_id)
            if decimal <= 0:
                raise ValueError("CoNLL-U decimal IDs have to be greater than 0")
            return decimal
    if isinstance(token_id, float):
        if token_id <= 0:
            raise ValueError("CoNLL-U decimal IDs have to be greater than 0")
        return Decimal(token_id)

    return token_id


def parse_key_value(comment: str) -> tuple[str, str] | str:
    """
    Parses a CoNLL-U comment line into a key-value pair if possible and returns the input string otherwise.

    Args:
        comment: Comment line from CoNLL-U format.

    Returns:
        (key, value) tuple for key=value format, or a plain string
        otherwise.
    """

    match = re.fullmatch(r"#\s*([^=]+?)\s*=\s*(.+?)\s*", comment)
    if match is None:
        return comment.removeprefix("#").lstrip()

    key, value = match.groups()
    return key, value


@dataclass(slots=True)
class CoNLLU:
    """
    CoNLL-U format with 10 standard columns. For reference, see: https://universaldependencies.org/format.html

    Attributes:
        id: Token ID (int, Decimal for empty nodes, or range for multi-
            word tokens).
        form: Token text.
        lemma: Lemma or stem.
        upos: Universal part-of-speech tag.
        xpos: Language-specific part-of-speech tag.
        feats: Dictionary of morphological features.
        head: Head token ID.
        deprel: Dependency relation.
        deps: Enhanced dependency graph in the form of a list of head-deprel pairs.
        misc: Dictionary of other miscellaneous information.
    """

    id: Annotated[int | Decimal | range, PlainValidator(conllu_id)]
    # Handle literal underscores as just underscores
    # Making this decision is left to the implementation according to https://universaldependencies.org/format.html
    form: str
    lemma: str
    upos: MaybeUnderscore
    xpos: MaybeUnderscore
    feats: Annotated[dict[str, str] | None, key_value_list(), maybe_empty()]
    head: Annotated[int | None, maybe_empty()]
    deprel: MaybeUnderscore
    deps: Annotated[dict[int, str] | None, key_value_list(key_value_sep=":"), maybe_empty()]
    # Handle SpaceAfter=No
    misc: Annotated[dict[str, str] | None, key_value_list(), maybe_empty()]

    @staticmethod
    def parse_comments(comments: list[str]) -> CoNLLMetadata:
        """
        Parse CoNLL-U style comments into CoNLLMetadata.

        Args:
            comments: List of comment strings starting with "#".

        Returns:
            Parsed CoNLLMetadata.
        """

        return CoNLLMetadata.with_key_value(parse_key_value(comment) for comment in comments)

    @staticmethod
    def is_document_start(metadata: CoNLLMetadata) -> bool:
        """
        Check if the metadata indicates the start of a new document.

        Args:
            metadata: The CoNLLMetadata to check.

        Returns:
            `True` if a "newdoc id" is present, indicating the start of
            a new document.
        """

        return metadata.get_meta("newdoc id") is not None


@CoNLLDialectRegistry.register("conllu_plus")
@dataclass(slots=True)
class CoNLLUPlus:
    """
    CoNLL-U Plus format with NER tags used by the NYTK-NerKor dataset. For reference, see: https://universaldependencies.org/ext-format.html

    Attributes:
        form: The token text.
        lemma: Lemma or stem.
        upos: Universal part-of-speech tag.
        xpos: Language-specific part-of-speech tag.
        feats: Dictionary of morphological features.
        ner: The NER tag in BIO format.
        emmorph_lemma: Lemma derived from the Hungarian emMorph
            morphological analyzer
    """

    form: str
    lemma: str
    upos: MaybeUnderscore
    xpos: MaybeUnderscore
    feats: Annotated[dict[str, str] | None, key_value_list(), maybe_empty()]
    ner: BIOField
    emmorph_lemma: str

    @staticmethod
    def parse_comments(comments: list[str]) -> CoNLLMetadata:
        """
        Parse CoNLL-U style comments into CoNLLMetadata.

        Args:
            comments: List of comment strings starting with "#".

        Returns:
            Parsed CoNLLMetadata.
        """

        return CoNLLMetadata.with_key_value(parse_key_value(comment) for comment in comments)

    @staticmethod
    def is_document_start(_: CoNLLMetadata) -> bool:
        """
        Check if the metadata indicates the start of a new document.

        Returns:
            Always returns True for CoNLLUPlus.
        """

        return True


def _maybe_multi_label(maybe_bio: str | None) -> list[BIO] | None:
    """
    Parse optional multi-label NER tags separated by hash.

    Args:
        maybe_bio: Optional NER string with hash-separated tags.

    Returns:
        List of BIO instances or `None` if input is `None`.
    """

    if maybe_bio is None:
        return None

    # Hash separated original NER only occurs in UNER Danish DDT
    return list(map(BIO.from_string, maybe_bio.split("#")))


@CoNLLDialectRegistry.register("conllu_uner")
@dataclass(slots=True)
class UNERCoNLLU:
    """
    Universal NER CoNLL-U-style format with UNER tags and (optionally) original NER tags.

    Attributes:
        id: Token ID.
        form: The token text.
        ner: The NER tag in BIO format.
        original_ner: Original NER tags for converted datasets.
        annotator: Optional name of the annotator.
    """

    id: int | Decimal
    form: str
    ner: BIOField
    original_ner: Annotated[list[BIO] | None, BeforeValidator(_maybe_multi_label), maybe_empty("-")]
    annotator: MaybeHyphen

    @staticmethod
    def parse_comments(comments: list[str]) -> CoNLLMetadata:
        """
        Parse CoNLL-U style comments into CoNLLMetadata.

        Args:
            comments: List of comment strings starting with "#".

        Returns:
            Parsed CoNLLMetadata.
        """

        return CoNLLMetadata.with_key_value(parse_key_value(comment) for comment in comments)

    @staticmethod
    def is_document_start(metadata: CoNLLMetadata) -> bool:
        """
        Check if the metadata indicates the start of a new document.

        Args:
            metadata: The CoNLLMetadata to check.

        Returns:
            `True` if a "newdoc id" is present, indicating the start of
            a new document.
        """

        return metadata.get_meta("newdoc id") is not None


@CoNLLDialectRegistry.register("conllu_uner_newpar")
@dataclass(slots=True)
class UNERCoNLLUNewPar(UNERCoNLLU):
    """
    Variant of the Universal NER CoNLL-U-style format using `newpar` comments as a document separator

    Attributes:
        id: Token ID.
        form: The token text.
        ner: The NER tag in BIO format.
        original_ner: Original NER tags for converted datasets.
        annotator: Optional name of the annotator.
    """

    @staticmethod
    def is_document_start(metadata: CoNLLMetadata) -> bool:
        """
        Check if the metadata indicates the start of a new document.

        Args:
            metadata: The CoNLLMetadata to check.

        Returns:
            `True` if a "newpar" comment is present, indicating the start of
            a new document.
        """

        return any(comment.strip() == "newpar" for comment in metadata.comments)


@CoNLLDialectRegistry.register("ner_suite")
@dataclass(slots=True)
class NerSuiteCoNLL:
    """
    NERSuite CoNLL-style format with character indices mapping tokens to the source text. For reference, see: https://nersuite.nlplab.org/advanced_usage.html

    Attributes:
        ner: The NER tag in BIO format.
        start: Start character offset of the token in the source text.
        end: End character offset of the token in the source text.
        form: The token text.
        lemma: Lemma or stem.
        pos: Part-of-speech tag.
        chunk: Chunk tag in BIO format.
    """

    ner: BIOField
    start: int
    end: int
    form: str
    lemma: str
    pos: MaybeUnderscore
    chunk: BIOField


@CoNLLDialectRegistry.register("conll2003_pioner")
@dataclass(slots=True)
class CoNLL2003Pioner:
    """
    CoNLL2003-style format with underscore placeholders and no -DOCSTART- for the pioNER dataset.

    Attributes:
        form: The token text.
        pos: Part-of-speech tag.
        syntactic_chunk: Syntactic chunk tag in BIO format.
        ner: The NER tag in BIO format.
    """

    form: str
    pos: MaybeUnderscore
    syntactic_chunk: Annotated[
        BIO | None,
        BeforeValidator(BIO.from_optional_string),
        maybe_empty(),
        PlainSerializer(str, when_used="unless-none"),
    ]
    ner: BIOField


@CoNLLDialectRegistry.register("conll2003")
@dataclass(slots=True)
class CoNLL2003:
    """
    Standard CoNLL2003 format with -DOCSTART- handling.

    Attributes:
        docstart: Class variable indicating the document start marker.
        ignore_docstart: Class variable indicating whether to ignore
            document start markers.
        form: The token form/word.
        pos: Part-of-speech tag.
        syntactic_chunk: Syntactic chunk tag in BIO format.
        ner: The Named Entity Recognition tag in BIO format.
    """

    docstart: ClassVar[str] = "-DOCSTART-"
    ignore_docstart: ClassVar[bool] = False

    form: str
    pos: MaybeHyphen
    syntactic_chunk: Annotated[
        BIO | None,
        BeforeValidator(BIO.from_optional_string),
        maybe_empty("-"),
        PlainSerializer(str, when_used="unless-none"),
    ]
    ner: BIOField


@CoNLLDialectRegistry.register("conll2003_ignore_docstart")
@dataclass(slots=True)
class CoNLL2003IgnoreDocstart(CoNLL2003):
    """
    CoNLL2003 format that ignores -DOCSTART- markers.

    Attributes:
        ignore_docstart: Class variable indicating whether to ignore
            document start markers.
    """

    ignore_docstart: ClassVar[bool] = True


@CoNLLDialectRegistry.register("conll2003_two_column")
@dataclass(slots=True)
class CoNLL2003TwoColumn:
    """
    Two-column CoNLL2003-style format with only word form and NER columns.

    Attributes:
        docstart: Class variable indicating the document start marker.
        ignore_docstart: Class variable indicating whether to ignore
            document start markers.
        form: The token text.
        ner: The NER tag in BIO format.
    """

    docstart: ClassVar[str] = "-DOCSTART-"
    ignore_docstart: ClassVar[bool] = False

    form: str
    ner: BIOField


@CoNLLDialectRegistry.register("conll_jnlpba")
@dataclass(slots=True)
class CoNLLJNLPBA:
    """
    CoNLL format for JNLPBA dataset with MEDLINE document markers.

    Attributes:
        docstart: Class variable indicating the document start marker.
        ignore_docstart: Class variable indicating whether to ignore
            document start markers.
        form: The token text.
        ner: The NER tag in BIO format.
    """

    docstart: ClassVar[str] = "###MEDLINE:"
    ignore_docstart: ClassVar[bool] = False

    form: str
    ner: BIOField


@CoNLLDialectRegistry.register("conll_stackoverflow_ner")
@dataclass(slots=True)
class CoNLLStackOverflowNER:
    """
    CoNLL format for the StackOverflow-NER dataset.

    Attributes:
        docstart: Class variable indicating the document start marker.
        ignore_docstart: Class variable indicating whether to ignore
            document start markers.
        form: The token text.
        ner: The NER tag in BIO format.
        form_2: Secondary form field.
        markdown: Markdown syntax metadata in BIO format.
    """

    docstart: ClassVar[str] = "-DOCSTART-"
    ignore_docstart: ClassVar[bool] = False

    form: str
    ner: BIOField
    form_2: str
    markdown: BIOField


@CoNLLColumnarRegistry.register("flat")
@dataclass(slots=True)
class FlatIndices:
    """
    CoNLL-style columnar format for data without blank lines, indicating documents and sentences via index columns.

    Attributes:
        ner: NER label.
        form: The token text.
        doc_idx: Document index.
        sent_idx: Sentence index.
    """

    ner: int
    form: str
    doc_idx: int
    sent_idx: int

    @staticmethod
    def segment_columns(rows: Iterable[dict[str, Any]]) -> Iterable[list[dict[str, Any]]]:
        """
        Segment flat index rows into sentence-level groups.

        Args:
            rows: Iterable of flat index rows.

        Returns:
            Iterable of sentence-level row groups.
        """

        current_index = 0
        sentence_rows = []
        for row in rows:
            index = row["sent_idx"]
            if index != current_index:
                current_index = index
                yield sentence_rows
                sentence_rows = []

            sentence_rows.append(row)

        if sentence_rows:
            yield sentence_rows


def space_separated_segments(lines: Iterable[str], enforce_blank_lines: bool = True) -> Iterator[list[str]]:
    """
    Split lines into segments separated by blank lines.

    Args:
        lines: Iterable of line strings.
        enforce_blank_lines: Whether to enforce blank lines as segment
            separators.
    Yields:
        Segments as lists of lines.
    """

    allow_whitespace_in_blank = not enforce_blank_lines
    current_segment = []
    for line in lines:
        line = line.strip("\r\n")
        if not line or (allow_whitespace_in_blank and not line.strip()):
            if current_segment:
                yield current_segment
                current_segment = []
        else:
            current_segment.append(line)

    # Handle final segment for non-compliant files that lack a final blank line
    if current_segment:
        yield current_segment


type RowType = CommonRowType | ColumnarCoNLL


@dataclass(slots=True)
class Sentence[T: RowType]:
    """
    A parsed CoNLL-style sentence with optional metadata.

    **Type Parameters:**

    T
        The type of rows in the sentence.

    Attributes:
        rows: List of parsed CoNLL-style rows.
        meta: Optional CoNLLMetadata for the sentence.
    """

    rows: list[T]
    meta: CoNLLMetadata | None = None


def parse_columns[C: ColumnarCoNLL](rows: Iterable[dict[str, Any]], dialect: type[C]) -> Iterator[Sentence[C]]:
    """
    Parse columnar rows into sentences using the specified CoNLL dialect.

    **Type Parameters:**

    C
        The CoNLL dialect class used for parsing rows.

    Args:
        rows: Iterable of row dictionaries.
        dialect: The CoNLL dialect class to use for parsing.
    Yields:
        Parsed sentences.
    """

    row_parser = TypeAdapter(dialect)

    for segment in dialect.segment_columns(rows):
        yield Sentence([row_parser.validate_python(row) for row in segment])


class RowParser[T]:
    """
    Parser for CoNLL-style rows supporting arbitrary CoNLL dialects.

    **Type Parameters:**

    T
        The CoNLL dialect class used for parsing rows.

    Args:
        dialect: The CoNLL dialect class to parse rows with.
    """

    def __init__(self, dialect: type[T]) -> None:
        self.row_parser = TypeAdapter(dialect)
        fields = dataclasses.fields(dialect)  # type: ignore
        self.fields = [field.name for field in fields]

    def validate_row(self, row: list[str]) -> T:
        """
        Validate and parse a row using the parser's CoNLL dialect.

        Args:
            row: List of fields in the row.

        Returns:
            Parsed Sentence using the parser's CoNLL dialect.
        """

        return self.row_parser.validate_python({field: value for field, value in zip(self.fields, row)})


def _parse_with_comments[T: type[CommentedCoNLL]](
    lines: Iterable[str],
    dialect: T,
    delimiter: str = "\t",
    enforce_blank_lines: bool = True,
    use_comment_document_boundary: bool = True,
) -> Iterator[list[Sentence[T]]]:
    if isinstance(dialect, DocumentSeparatedCoNLL):
        raise NotImplementedError("Combining -DOCSTART- with CoNLL-U style comments is currently unsupported")

    parser = RowParser(dialect)

    document_sentences = []
    for segment in space_separated_segments(lines, enforce_blank_lines):
        comments = []
        end_of_comment = 0
        for i, line in enumerate(segment):
            if not line.startswith("#"):
                end_of_comment = i
                break
            comments.append(line)

        sentence = Sentence(
            [parser.validate_row(line.split(delimiter)) for line in segment[end_of_comment:]],
            dialect.parse_comments(comments),
        )

        if sentence.meta is None:
            raise ValueError("Encountered commented CoNLL sentence without metadata")

        # End the previous document if necessary and initialize a new one unless disabled
        if not use_comment_document_boundary:
            yield [sentence]
        elif dialect.is_document_start(sentence.meta) and document_sentences:
            yield document_sentences
            document_sentences = [sentence]
        else:
            document_sentences.append(sentence)

    # Handle trailing document
    if document_sentences:
        yield document_sentences


def parse[T: RowType](
    lines: Iterable[str],
    dialect: type[T] = CoNLL,
    delimiter: str = "\t",
    enforce_blank_lines: bool = True,
    use_comment_document_boundary: bool = True,
) -> Iterator[list[Sentence[T]]]:
    """
    Parse CoNLL-style lines into documents with sentences.

    **Type Parameters:**

    T
        The CoNLL dialect class used for parsing rows.

    Args:
        lines: Iterable of line strings.
        dialect: The CoNLL dialect class to use for parsing.
        delimiter: Field delimiter separating the CoNLL-style columns.
        enforce_blank_lines: Whether to enforce that blank lines between
            segments are empty. Otherwise, lines that contain only
            whitespace will be treated as blank
        use_comment_document_boundary: Whether to parse document
            boundaries from CoNLL-U-style comment.

    Returns:
        Iterator over documents (lists of sentences).

    Raises:
        NotImplementedError: When combining -DOCSTART- with CoNLL-U
            style comments.
    """

    if isinstance(dialect, CommentedCoNLL):
        yield from _parse_with_comments(lines, dialect, delimiter, use_comment_document_boundary)
        return

    parser = RowParser(dialect)

    if isinstance(dialect, DocumentSeparatedCoNLL):
        docstart = dialect.docstart
        ignore_docstart = dialect.ignore_docstart
    else:
        docstart = None
        ignore_docstart = True

    document_sentences = []
    for segment in space_separated_segments(lines, enforce_blank_lines):
        if docstart is not None and segment[0].startswith(docstart):
            # Ignore and skip -DOCSTART- lines
            if ignore_docstart:
                continue

            # Avoids yielding an empty sentence at the start of the file
            if document_sentences:
                yield document_sentences
                document_sentences = []

            # Skip docstart only segments entirely and otherwise remove the docstart header
            if len(segment) == 1:
                continue

            segment = segment[1:]

        sentence = Sentence([parser.validate_row(line.split(delimiter)) for line in segment])

        if ignore_docstart:
            yield [sentence]
        else:
            document_sentences.append(sentence)

    # Handle trailing document
    if document_sentences:
        yield document_sentences
