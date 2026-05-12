"""Tests for the data pipes system."""

import pytest

from meld.data_pipes import DataPipe, DataPipeError
from meld.manifest import (
    CoNLLConfiguration,
    ConvertStep,
    DataPipeStep,
    DownloadStep,
    ExtractStep,
    HuggingfaceArguments,
    HuggingfaceLoader,
    ReaderConfigurationWithoutArguments,
    SplitStep,
    URLWithChecksum,
    load_manifest,
)


def test_manifest():
    # Tests whether dataset configurations can be loaded successfully
    load_manifest()


def test_data_pipe_validation():
    # Test with a valid data pipe
    steps = [
        DownloadStep(step="download", urls=[URLWithChecksum("http://example.com/file.zip", "SHA")]),
        ExtractStep(step="extract", from_file="file.zip", files=["file1.txt", "file2.txt"]),
        SplitStep(step="splits", language="eng", train=["file1.txt"], test=["file2.txt"]),
        ConvertStep(step="convert", reader=CoNLLConfiguration("conll")),
    ]
    data_pipe = DataPipe(steps)
    assert len(data_pipe._stages) == 4
    assert all(len(stage) == 1 for stage in data_pipe._stages)

    # Test with a missing dependency
    steps_missing_dependency = [
        SplitStep(
            step="splits", language="eng", train=["train1.txt"], validation=["validation1.txt"], test=["test1.txt"]
        ),
        ConvertStep(step="convert", reader=CoNLLConfiguration("conll")),
    ]
    with pytest.raises(DataPipeError):
        data_pipe = DataPipe(steps_missing_dependency)


def test_data_pipe_multi_stage():
    # Test with a valid data_pipe
    steps = [
        DownloadStep(step="download", urls=[URLWithChecksum("http://example.com/file.zip", "SHA_file1")]),
        DownloadStep(step="download", urls=[URLWithChecksum("http://example.com/file2.zip", "SHA_file2")]),
        ExtractStep(step="extract", from_file="file.zip", files=["file1.txt", "file2.txt"]),
        ExtractStep(step="extract", from_file="file2.zip", files=["file3.txt", "file4.txt"]),
        SplitStep(step="splits", language="eng", train=["file1.txt", "file3.txt", "file4.txt"], test=["file2.txt"]),
        ConvertStep(step="convert", reader=CoNLLConfiguration("conll")),
    ]
    data_pipe = DataPipe(steps)
    assert len(data_pipe._stages) == 4
    for step, length in enumerate((2, 2, 1, 1)):
        assert len(data_pipe._stages[step]) == length

    # Test with an invalid duplicate
    steps_duplicate_split = [
        DownloadStep(
            step="download",
            urls=[
                URLWithChecksum("http://example.com/file.zip", "SHA_file1"),
                URLWithChecksum("http://example.com/file2.zip", "SHA_file2"),
            ],
        ),
        ExtractStep(step="extract", from_file="file.zip", files=["file1.txt", "file2.txt"]),
        ExtractStep(step="extract", from_file="file2.zip", files=["file3.txt", "file4.txt"]),
        SplitStep(step="splits", language="eng", train=["file1.txt", "file3.txt"], test=["file2.txt"]),
        SplitStep(step="splits", language="eng", test=["file4.txt"]),
        ConvertStep(step="convert", reader=CoNLLConfiguration("conll")),
        ConvertStep(step="convert", reader=ReaderConfigurationWithoutArguments("bioc_xml")),
    ]
    with pytest.raises(DataPipeError):
        data_pipe = DataPipe(steps_duplicate_split)


def test_data_pipe_loader_validation():
    # Test with a valid loader
    steps: list[DataPipeStep] = [ConvertStep(step="convert", reader=CoNLLConfiguration("conll"))]
    data_pipe = DataPipe(
        steps,
        loader=HuggingfaceLoader(
            "huggingface",
            HuggingfaceArguments("example_repo", "revision", "train", "validation", "test", "text", "tags"),
        ),
    )
    assert len(data_pipe._stages) == 2
    assert len(data_pipe._stages[0]) == 1
    assert len(data_pipe._stages[1]) == 1

    # Test with a loader that requires extra dependencies
    with pytest.raises(DataPipeError):
        data_pipe = DataPipe(steps)
