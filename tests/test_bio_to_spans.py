"""Tests for the meld.tokenization.bio_to_spans function."""

from meld.document import BIO, Annotation, Span
from meld.tokenization import bio_to_spans


def test_bio_to_spans_simple_single_entity():
    """Test simple single entity spanning multiple tokens."""
    bio_labels = [
        BIO.from_string("O"),
        BIO.from_string("B-person"),
        BIO.from_string("I-person"),
        BIO.from_string("I-person"),
        BIO.from_string("O"),
    ]
    token_spans = [
        Span(0, 5),
        Span(5, 10),
        Span(10, 15),
        Span(15, 20),
        Span(20, 21),
    ]

    result = bio_to_spans(bio_labels, token_spans)

    expected = [Annotation.from_span("person", 5, 20)]
    assert result == expected


def test_bio_to_spans_multiple_entities():
    """Test multiple separate entities in the same input."""
    bio_labels = [
        BIO.from_string("B-person"),
        BIO.from_string("I-person"),
        BIO.from_string("O"),
        BIO.from_string("B-organization"),
        BIO.from_string("I-organization"),
    ]
    token_spans = [
        Span(0, 5),
        Span(5, 10),
        Span(10, 11),
        Span(11, 16),
        Span(16, 21),
    ]

    result = bio_to_spans(bio_labels, token_spans)

    expected = [
        Annotation.from_span("person", 0, 10),
        Annotation.from_span("organization", 11, 21),
    ]
    assert result == expected


def test_bio_to_spans_consecutive_same_label():
    """Test consecutive entities with same label are merged."""
    bio_labels = [
        BIO.from_string("B-person"),
        BIO.from_string("I-person"),
        BIO.from_string("O"),
        BIO.from_string("B-person"),
        BIO.from_string("I-person"),
    ]
    token_spans = [
        Span(0, 5),
        Span(5, 10),
        Span(10, 11),
        Span(11, 16),
        Span(16, 21),
    ]

    result = bio_to_spans(bio_labels, token_spans)

    expected = [
        Annotation.from_span("person", 0, 10),
        Annotation.from_span("person", 11, 21),
    ]
    assert result == expected


def test_bio_to_spans_all_o_tags():
    """Test input with only O tags (no entities)."""
    bio_labels = [
        BIO.from_string("O"),
        BIO.from_string("O"),
        BIO.from_string("O"),
    ]
    token_spans = [
        Span(0, 4),
        Span(4, 8),
        Span(8, 12),
    ]

    result = bio_to_spans(bio_labels, token_spans)

    expected = []
    assert result == expected


def test_bio_to_spans_single_token_entities():
    """Test single token entities (B tag followed by B tag or end)."""
    bio_labels = [
        BIO.from_string("O"),
        BIO.from_string("B-person"),
        BIO.from_string("B-organization"),
        BIO.from_string("O"),
        BIO.from_string("B-organization"),
    ]
    token_spans = [
        Span(0, 1),
        Span(1, 4),
        Span(4, 8),
        Span(8, 12),
        Span(12, 15),
        Span(15, 18),
    ]

    result = bio_to_spans(bio_labels, token_spans)

    expected = [
        Annotation.from_span("person", 1, 4),
        Annotation.from_span("organization", 4, 8),
        Annotation.from_span("organization", 12, 15),
    ]
    assert result == expected


def test_bio_to_spans_empty_input():
    """Test empty input lists."""
    bio_labels: list[BIO] = []
    token_spans: list[Span] = []

    result = bio_to_spans(bio_labels, token_spans)

    expected = []
    assert result == expected


def test_bio_to_spans_span_boundary_handling():
    """Test span boundary calculations with various span lengths."""
    bio_labels = [
        BIO.from_string("B-person"),
        BIO.from_string("I-person"),
        BIO.from_string("O"),
        BIO.from_string("B-organization"),
    ]
    token_spans = [
        Span(0, 2),
        Span(2, 4),
        Span(4, 5),
        Span(5, 8),
    ]

    result = bio_to_spans(bio_labels, token_spans)
    expected = [
        Annotation.from_span("person", 0, 4),
        Annotation.from_span("organization", 5, 8),
    ]
    assert result == expected

    bio_labels = [
        BIO.from_string("O"),
        BIO.from_string("B-location"),
        BIO.from_string("O"),
    ]
    token_spans = [
        Span(0, 1),
        Span(1, 3),
        Span(3, 4),
    ]

    result = bio_to_spans(bio_labels, token_spans)
    expected = [Annotation.from_span("location", 1, 3)]
    assert result == expected

    bio_labels = [
        BIO.from_string("B-title"),
        BIO.from_string("I-title"),
        BIO.from_string("O"),
        BIO.from_string("B-person"),
    ]
    token_spans = [
        Span(0, 8),
        Span(8, 9),
        Span(9, 10),
        Span(10, 15),
    ]

    result = bio_to_spans(bio_labels, token_spans)
    expected = [
        Annotation.from_span("title", 0, 9),
        Annotation.from_span("person", 10, 15),
    ]
    assert result == expected


def test_bio_to_spans_spanning_borders():
    """Test entities that span across character boundaries correctly."""
    bio_labels = [
        BIO.from_string("B-person"),
        BIO.from_string("I-person"),
        BIO.from_string("I-person"),
        BIO.from_string("O"),
    ]
    token_spans = [
        Span(0, 3),
        Span(3, 6),
        Span(6, 9),
        Span(9, 10),
    ]

    result = bio_to_spans(bio_labels, token_spans)

    expected = [Annotation.from_span("person", 0, 9)]
    assert result == expected


def test_bio_to_spans_single_token_entity_at_start():
    """Test single token entity at the start of input."""
    bio_labels = [
        BIO.from_string("B-person"),
        BIO.from_string("O"),
        BIO.from_string("O"),
    ]
    token_spans = [
        Span(0, 5),
        Span(5, 10),
        Span(10, 15),
    ]

    result = bio_to_spans(bio_labels, token_spans)

    expected = [Annotation.from_span("person", 0, 5)]
    assert result == expected


def test_bio_to_spans_single_token_entity_at_end():
    """Test single token entity at the end of input."""
    bio_labels = [
        BIO.from_string("O"),
        BIO.from_string("O"),
        BIO.from_string("B-person"),
    ]
    token_spans = [
        Span(0, 3),
        Span(3, 6),
        Span(6, 11),
    ]

    result = bio_to_spans(bio_labels, token_spans)

    expected = [Annotation.from_span("person", 6, 11)]
    assert result == expected


def test_bio_to_spans_different_entity_types():
    """Test multiple different entity types."""
    bio_labels = [
        BIO.from_string("B-person"),
        BIO.from_string("B-organization"),
        BIO.from_string("B-location"),
    ]
    token_spans = [
        Span(0, 5),
        Span(5, 9),
        Span(9, 13),
    ]

    result = bio_to_spans(bio_labels, token_spans)

    expected = [
        Annotation.from_span("person", 0, 5),
        Annotation.from_span("organization", 5, 9),
        Annotation.from_span("location", 9, 13),
    ]
    assert result == expected


def test_bio_to_spans_long_entity_sequence():
    """Test long sequence of I tags following a B tag."""
    bio_labels = [
        BIO.from_string("B-product"),
        BIO.from_string("I-product"),
        BIO.from_string("I-product"),
        BIO.from_string("I-product"),
        BIO.from_string("I-product"),
        BIO.from_string("I-product"),
        BIO.from_string("O"),
    ]
    token_spans = [
        Span(0, 4),
        Span(4, 8),
        Span(8, 12),
        Span(12, 16),
        Span(16, 20),
        Span(20, 24),
        Span(24, 25),
    ]

    result = bio_to_spans(bio_labels, token_spans)

    expected = [Annotation.from_span("product", 0, 24)]
    assert result == expected


def test_bio_to_spans_tags_without_entity_type():
    """Test that B/I tags without entity type raise ValueError."""
    bio_labels = [
        BIO("B"),
        BIO.from_string("O"),
    ]
    token_spans = [
        Span(0, 4),
        Span(4, 7),
    ]

    try:
        bio_to_spans(bio_labels, token_spans)
        assert False, "Expected ValueError was not raised"
    except ValueError:
        pass


def test_bio_to_spans_i_without_b():
    """Test I tag without preceding B tag behavior."""
    bio_labels = [
        BIO.from_string("O"),
        BIO.from_string("I-person"),
        BIO.from_string("O"),
    ]
    token_spans = [
        Span(0, 1),
        Span(1, 6),
        Span(6, 9),
    ]

    result = bio_to_spans(bio_labels, token_spans)

    # The I-person tag at position 1 is treated as starting a new entity
    # because it has a type but current_tag is None. This means it behaves
    # like a B tag.

    expected = [Annotation.from_span("person", 1, 6)]
    assert result == expected
