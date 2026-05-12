"""Data statistics and analysis utilities."""

import logging
import multiprocessing
from collections.abc import Iterator
from dataclasses import dataclass

import langcodes
import regex

try:
    from datatrove.utils import word_tokenizers
    from datatrove.utils.word_tokenizers import StanzaTokenizer, WordTokenizer
except ImportError:
    word_tokenizers = None
    StanzaTokenizer = None

from meld.formats import Dataset, Subset

logger = logging.getLogger("meld")
_IGNORED_LANGUAGES = {
    "MULTI"  # Subset of MultiCoNER II without language metadata
}


@dataclass(slots=True)
class WordSequenceStats:
    """
    Statistics for a tokenized sequence.

    Attributes:
        dataset_name: Name of the dataset.
        subset_hierarchy: Subset hierarchy of the split.
        split_name: Name of the data split.
        language: Language code.
        sequence_id: Unique identifier for the sequence.
        document_index: Index of the document in the dataset.
        document_position: Position index of a sentence or paragraph
            within the document.
        total_token_count: Frequency of all tokens.
        number_count: Frequency of numeric tokens.
        punctuation_count: Frequency of punctuation tokens.
        word_count: Frequency of words (excluding numbers and
            punctuation).
    """

    dataset_name: str
    subset_hierarchy: list[str]
    split_name: str
    language: str
    sequence_id: bytes
    document_index: int
    document_position: int

    total_token_count: int
    number_count: int
    punctuation_count: int
    word_count: int


tokenizer = None


# Fall back to Mandarin where no specific tokenizers are available
_FINEWEB_2_MAPPING = {"zho": "cmn", "nan": "cmn"}
# Langcodes would produce kor_Kore instead of kor_Hang
_FINEWEB_2_SCRIPTS = {"kor": "kor_Hang"}


def _langcode_to_fineweb2(langcode: str) -> str:
    """
    Converts language codes to FineWeb2 format.

    Args:
        langcode: ISO 639-3 language code.

    Returns:
        Language code following the FineWeb2 subset naming scheme with
        an inferred script suffix if possible.
    """
    # Language codes that require special handling
    if langcode in _FINEWEB_2_SCRIPTS:
        return _FINEWEB_2_SCRIPTS[langcode]

    langcode = _FINEWEB_2_MAPPING.get(langcode, langcode)
    inferred_script = langcodes.get(langcode).assume_script().script
    if inferred_script is None:
        return langcode

    # Add script to the ISO 639-3 tag with an underscore separator to match the expected FineWeb2 format
    return f"{langcode}_{inferred_script}"


def _try_loading_tokenizer(language: str) -> "WordTokenizer | None":
    """
    Attempts to load a word tokenizer for the given language.

    Args:
        language: Language code to load tokenizer for. Converted to
            FineWeb2 format before loading.

    Returns:
        WordTokenizer instance if available, None otherwise.
    """

    if word_tokenizers is None:
        raise ValueError("datatrove needs to be installed for word tokenziation support")

    try:
        language = _langcode_to_fineweb2(language)
        logger.info(f"Attempting to load tokenizer for {language!r}")
        tokenizer = word_tokenizers.load_word_tokenizer(language)
        logger.info(f"Using {tokenizer.language} tokenizer for {language}")
    except ValueError:
        return None

    return tokenizer


def _load_tokenizer(language: str) -> None:
    """
    Loads and stores a word tokenizer globally within the current process for the given language.

    Args:
        language: Language code to load tokenizer for.
    """

    global tokenizer

    tokenizer = _try_loading_tokenizer(language)


@dataclass(slots=True)
class _CollectSplitsArguments:
    """
    Arguments for tokenizing and collecting statistics for a single sequence.

    Attributes:
        dataset_name: Name of the dataset.
        subset_hierarchy: Subset hierarchy of the split.
        split_name: Name of the data split.
        language: Language code of the split.
        sequence_id: Unique identifier for the sequence.
        document_index: Index of the document in the dataset.
        document_position: Position index of a sentence or paragraph
            within its containing document.
        sequence: The text or token list to process. May be a string
            requiring tokenization or a pre-tokenized list of tokens.
    """

    dataset_name: str
    subset_hierarchy: list[str]
    split_name: str
    language: str
    sequence_id: bytes
    document_index: int
    document_position: int
    sequence: list[str] | str


_punctuation_pattern = regex.compile(r"^\p{P}+$")
_number_pattern = regex.compile(r"^\p{P}*\p{N}[\p{N}\p{P}]*$")


def _tokenize_sequence(arguments: _CollectSplitsArguments) -> WordSequenceStats:
    """
    Tokenizes a single input text and computes word frequency statistics.

    Uses a word tokenizer if available for string inputs. Otherwise, falls back to using a
    pre-tokenized version of the text if available. Counts total tokens, punctuation, numbers,
    and words.

    Args:
        arguments: A text to tokenize with metadata.

    Returns:
        WordSequenceStats with token frequency counts for the sequence.
    """

    if tokenizer is not None and isinstance(arguments.sequence, str):
        tokens = tokenizer.word_tokenize(arguments.sequence)
    else:
        tokens = arguments.sequence

    total_token_count = 0
    punctuation_count = 0
    number_count = 0

    for token in tokens:
        total_token_count += 1

        if _punctuation_pattern.match(token):
            punctuation_count += 1
        elif _number_pattern.match(token):
            number_count += 1

    return WordSequenceStats(
        arguments.dataset_name,
        arguments.subset_hierarchy,
        arguments.split_name,
        arguments.language,
        arguments.sequence_id,
        arguments.document_index,
        arguments.document_position,
        total_token_count,
        number_count,
        punctuation_count,
        total_token_count - number_count - punctuation_count,
    )


def _collect_split_token_stats(
    dataset_name: str, subset: Subset, split_name: str, workers: int = 12
) -> Iterator[WordSequenceStats]:
    """
    Collects token-level statistics for all documents in a dataset split.

    Loads the appropriate tokenizer for the split's language, processes sequences
    in parallel using multiprocessing, and yields WordSequenceStats with token counts
    (total, punctuation, numeric, and word tokens) for each sequence.

    Args:
        dataset_name: Name of the dataset.
        subset: Dataset subset to process.
        split_name: Name of the split to process.
        workers: Number of workers to use for parallel word tokenization.
            Note that a high worker count will increase memory consumption substantially for some tokenizers

    Yields:
        WordSequenceStats for each sequence.

    Raises:
        ValueError: If no tokenizer is available and pre-tokenized text
            is not present in the dataset, or if datatrove is not
            installed.
    """

    subset_hierarchy = subset.hierarchy
    language = subset.metadata.language
    if StanzaTokenizer is None:
        raise ValueError("datatrove needs to be installed for word tokenziation support")

    tokenizer = _try_loading_tokenizer(language)
    parquet_file = subset.open_split(split_name)

    if tokenizer is not None:
        columns = ["document_index", "document_position", "sequence_id", "text"]
        sequence_column = "text"
        # Workaround since Stanza Tokenizers using MWT might cause OOM with many threads
        if isinstance(tokenizer, StanzaTokenizer):
            workers = 1
    else:
        if "tokens" not in parquet_file.schema_arrow.names:
            raise ValueError(
                f"Could not load {language} tokenizer or pre-tokenized text to fall back to for subset {'.'.join(subset_hierarchy) or 'main'} of {dataset_name}"
            )

        logger.warning(
            f"Could not load {language} tokenizer for subset {'.'.join(subset_hierarchy) or 'main'} of {dataset_name}, falling back to pre-tokenized text"
        )
        columns = ["document_index", "document_position", "sequence_id", "tokens"]
        sequence_column = "tokens"

    with multiprocessing.Pool(workers, initializer=_load_tokenizer, initargs=(language,)) as pool:
        yield from pool.imap_unordered(
            _tokenize_sequence,
            (
                _CollectSplitsArguments(
                    dataset_name,
                    subset_hierarchy,
                    split_name,
                    language,
                    sequence_id,
                    document_index,
                    document_position,
                    sequence.as_py(),
                )
                for batch in parquet_file.iter_batches(columns=columns)
                for sequence_id, document_index, document_position, sequence in zip(
                    batch["sequence_id"], batch["document_index"], batch["document_position"], batch[sequence_column]
                )
            ),
            chunksize=1,
        )


def word_tokenize_dataset(dataset: Dataset, workers: int = 12) -> Iterator[WordSequenceStats]:
    """
    Tokenizes all sequences in a dataset and yields sequence-level word frequency statistics.

    Processes each subset and split, skipping ignored languages (e.g. MULTI). Logs progress after each subset.

    Args:
        dataset: Dataset to tokenize.
        workers: Number of workers to use for parallel word tokenization.
            Note that a high worker count will increase memory consumption substantially for some tokenizers

    Yields:
        WordSequenceStats for each sequence.
    """

    for subset in dataset:
        for split in subset.splits:
            if subset.metadata.language in _IGNORED_LANGUAGES:
                continue

            stats = _collect_split_token_stats(
                dataset.metadata.name,
                subset,
                split,
                workers,
            )

            yield from stats

        logger.info(f"Processed subset {dataset.metadata.name}->{'->'.join(subset.hierarchy)}")
