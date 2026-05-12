"""Data structures for representing NER annotations, tags, and datasets"""

from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, PlainSerializer

_BIOES_TO_BIO = {
    "B": "B",
    "I": "I",
    "O": "O",
    # End to inside since it always appears at the end of a multi-token span
    "E": "I",
    # Single to beginning since it only appears at the start of entities
    "S": "B",
}


@dataclass(slots=True, frozen=True)
class BIO:
    """
    BIO (Begin, Inside, Outside) tag format for Named Entity Recognition.

    Attributes:
        position: The position indicator (such as, B, I, or O). While
            "O" is always treated as "Outside", other positions than B
            and I are allowed to support expanded tagging schemes such
            as BIOES
        entity_type: The entity type, or None for O tags.
    """

    position: str
    entity_type: str | None = None

    @classmethod
    def from_string(cls, bio_string: str) -> Self:
        """
        Create a BIO instance from a string representation.

        Args:
            bio_string: The BIO string to parse (e.g., "B-PERS", "O").

        Returns:
            A parsed BIO instance.

        Raises:
            ValueError: If the BIO tag format is invalid.
        """

        if bio_string == "O":
            return cls("O")

        position, entity_type = bio_string.split("-", 1)
        if not (position and entity_type):
            raise ValueError(f"The prefix and type of a BIO tag must be non-empty: {bio_string}")
        return cls(position, entity_type)

    @classmethod
    def from_optional_string(cls, bio_string: str | None) -> Self | None:
        """
        Create a BIO instance from an optional string representation.

        Args:
            bio_string: The BIO string to parse, or None.

        Returns:
            A parsed BIO instance or None if bio_string is None.
        """

        return bio_string if bio_string is None else cls.from_string(bio_string)

    def bioes_to_bio(self) -> Self:
        """
        Convert a BIOES tag to BIO format by mapping "E" to "I" and "S" to "B".

        Returns:
            A new BIO instance in BIO format.
        """

        return self.__class__(_BIOES_TO_BIO[self.position], self.entity_type)

    def __str__(self) -> str:
        """
        Convert the BIO tag to its string representation.

        Returns:
            The string representation of the BIO tag (e.g., "B-PERS",
            "O").
        """

        return self.position if self.entity_type is None else f"{self.position}-{self.entity_type}"


type BIOField = Annotated[BIO, BeforeValidator(BIO.from_string), PlainSerializer(str)]


@dataclass(slots=True)
class LabeledTokens:
    """
    Tokens with token-level annotations.

    Attributes:
        tokens: List of tokens.
        labels: Mapping of tagset names to BIO tag lists.
        sequence_type: Whether the tokens represent a sentence or
            passage.
    """

    tokens: list[str]
    labels: dict[str, list[BIOField]]
    sequence_type: Literal["sentence", "passage"] = "sentence"


@dataclass(slots=True, frozen=True)
class Span:
    """
    A span with character-level start and stop indices.

    Attributes:
        start: Start index (inclusive).
        stop: Stop index (exclusive).
    """

    start: int
    stop: int

    def __len__(self) -> int:
        """
        Calculate the length of the span.

        Returns:
            The number of characters in the span (stop - start).
        """

        return self.stop - self.start

    @classmethod
    def from_run_length(cls, start: int, length: int) -> Self:
        """
        Create a span from a start index and length.

        Args:
            start: Start index.
            length: Length of the span.

        Returns:
            A new Span instance.
        """

        return cls(start, start + length)

    def __str__(self) -> str:
        """
        Create a string representation of the span.

        Returns:
            Span string representation formatted as "[start,stop]".
        """

        return f"[{self.start},{self.stop}]"


@dataclass(slots=True, frozen=True)
class Annotation:
    """
    An annotation with a label and associated spans.

    Attributes:
        label: The annotation label.
        spans: Tuple of Span objects. A sequence of more than one span
            represents a discontinuous span.
    """

    label: str
    spans: tuple[Span, ...]

    @classmethod
    def from_span(cls, label: str, start: int, stop: int) -> Self:
        """
        Create an annotation from a label and a single, contiguous span.

        Args:
            label: The annotation label.
            start: Start index of the span.
            stop: Stop index of the span.

        Returns:
            A new Annotation instance.
        """

        return cls(label, (Span(start, stop),))


type SequenceType = Literal["sentence", "passage"]

SEQUENCE_TYPES: dict[SequenceType, Literal[0, 1]] = {"sentence": 0, "passage": 1}


@dataclass(slots=True)
class LabeledText:
    """
    Unsegmented text with annotated spans.

    Attributes:
        text: The text content.
        labels: Mapping of tagsets to lists of annotations.
        sequence_type: Whether the text a single sentence or passage.
        space_after: Space characters occurring after the text in the
            source document.
    """

    text: str
    labels: dict[str, list[Annotation]]
    sequence_type: SequenceType = "sentence"
    space_after: str = ""


@dataclass(slots=True)
class NERDocument:
    """
    A document containing labeled text and optional BIO annotated tookens.

    Attributes:
        spans: List of labeled text spans.
        bio: Optional list of labeled tokens in BIO format.
    """

    spans: list[LabeledText]
    bio: list[LabeledTokens] | None = None
