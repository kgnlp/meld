"""Tests for tokenization and sentence splitting."""

from meld.conll import BIO
from meld.document import LabeledText, LabeledTokens, Span
from meld.formats import Annotation, NERDocument, default_tagset
from meld.tokenization import SentenceSplitter


def test_sentence_splitter():
    basic_splitter = SentenceSplitter(None, read_spans=False)
    actual_document = basic_splitter.sentence_tokenize(
        NERDocument(
            [
                LabeledText(
                    "This first sentence contains a Named Entity. This sentence contains a Named Entity.  That spans three sentences. This includes this one. This sentence also contains a Named Entity.",
                    default_tagset(
                        [
                            Annotation("general", (Span(32, 44),)),
                            Annotation("sentence_spanning", (Span(69, 136),)),
                            Annotation("general", (Span(168, 180),)),
                        ]
                    ),
                ),
            ],
            [
                LabeledTokens(
                    [
                        "This",
                        "first",
                        "sentence",
                        "contains",
                        "a",
                        "Named",
                        "Entity",
                        ".",
                        "This",
                        "sentence",
                        "contains",
                        "a",
                        "Named",
                        "Entity",
                        ".",
                        "That",
                        "spans",
                        "three",
                        "sentences",
                        ".",
                        "This",
                        "includes",
                        "this",
                        "one",
                        ".",
                        "This",
                        "sentence",
                        "also",
                        "contains",
                        "a",
                        "Named",
                        "Entity",
                        ".",
                    ],
                    default_tagset(
                        list(
                            map(
                                BIO.from_string,
                                [
                                    "O",
                                    "O",
                                    "O",
                                    "O",
                                    "O",
                                    "B-general",
                                    "I-general",
                                    "O",
                                    "O",
                                    "O",
                                    "O",
                                    "B-sequence_spanning",
                                    "I-sequence_spanning",
                                    "I-sequence_spanning",
                                    "I-sequence_spanning",
                                    "I-sequence_spanning",
                                    "I-sequence_spanning",
                                    "I-sequence_spanning",
                                    "I-sequence_spanning",
                                    "I-sequence_spanning",
                                    "I-sequence_spanning",
                                    "I-sequence_spanning",
                                    "I-sequence_spanning",
                                    "I-sequence_spanning",
                                    "O",
                                    "O",
                                    "O",
                                    "O",
                                    "O",
                                    "O",
                                    "B-general",
                                    "I-general",
                                    "O",
                                ],
                            )
                        ),
                    ),
                )
            ],
        )
    )

    expected_document = NERDocument(
        [
            LabeledText(
                "This first sentence contains a Named Entity.",
                {"ner": [Annotation(label="general", spans=(Span(start=32, stop=44),))]},
                space_after=" ",
            ),
            LabeledText(
                "This sentence contains a Named Entity.  That spans three sentences. This includes this one.",
                {"ner": [Annotation(label="sentence_spanning", spans=(Span(start=24, stop=91),))]},
                space_after=" ",
            ),
            LabeledText(
                "This sentence also contains a Named Entity.",
                {"ner": [Annotation(label="general", spans=(Span(start=31, stop=43),))]},
                space_after="",
            ),
        ],
        [
            LabeledTokens(
                ["This", "first", "sentence", "contains", "a", "Named", "Entity", "."],
                default_tagset(list(map(BIO.from_string, ["O", "O", "O", "O", "O", "B-general", "I-general", "O"]))),
            ),
            LabeledTokens(
                [
                    "This",
                    "sentence",
                    "contains",
                    "a",
                    "Named",
                    "Entity",
                    ".",
                    "That",
                    "spans",
                    "three",
                    "sentences",
                    ".",
                    "This",
                    "includes",
                    "this",
                    "one",
                    ".",
                ],
                default_tagset(
                    list(
                        map(
                            BIO.from_string,
                            [
                                "O",
                                "O",
                                "O",
                                "B-sequence_spanning",
                                "I-sequence_spanning",
                                "I-sequence_spanning",
                                "I-sequence_spanning",
                                "I-sequence_spanning",
                                "I-sequence_spanning",
                                "I-sequence_spanning",
                                "I-sequence_spanning",
                                "I-sequence_spanning",
                                "I-sequence_spanning",
                                "I-sequence_spanning",
                                "I-sequence_spanning",
                                "I-sequence_spanning",
                                "O",
                            ],
                        )
                    )
                ),
            ),
            LabeledTokens(
                ["This", "sentence", "also", "contains", "a", "Named", "Entity", "."],
                default_tagset(list(map(BIO.from_string, ["O", "O", "O", "O", "O", "B-general", "I-general", "O"]))),
            ),
        ],
    )

    assert actual_document == expected_document
