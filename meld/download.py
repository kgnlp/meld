"""Dataset download utilities."""

import contextlib
import logging
import shutil
import tarfile
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from email.message import Message
from io import BytesIO
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, overload
from urllib.parse import urlparse
from zipfile import ZipFile

import requests
from git import RemoteProgress, Repo
from tqdm import tqdm

logger = logging.getLogger("meld")


class _GitProgress(RemoteProgress):
    """Tracks the progress of a git operation with a `tqdm` progress bar."""

    def __init__(self):
        super().__init__()
        self._progress = tqdm()
        self._previous = 0
        self._last_op_code = None

    def update(
        self,
        op_code: int,
        cur_count: str | float,
        max_count: str | float | None = None,
        message: str = "",
    ) -> None:
        """
        Update the progress bar based on the current operation code and count.

        Args:
            op_code: The current operation code.
            cur_count: The current count of the operation.
            max_count: The maximum count of the operation.
            message: Unused
        """

        if isinstance(cur_count, str) or isinstance(max_count, str):
            return

        if op_code != self._last_op_code:
            self._previous = 0
            self._progress.reset(max_count)
            self._last_op_code = op_code
            if op_code == self.RESOLVING:
                self._progress.set_postfix_str("Resolving")

        self._progress.update(cur_count - self._previous)
        self._previous = cur_count

    # Only included for typing consistency with Repo.clone_from
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Call the update method with the provided arguments.

        Args:
            *args: Positional arguments to pass to the update method.
            **kwds: Keyword arguments to pass to the update method.
        """

        return self.update(*args, **kwds)


def git_download(
    repo_url: str, revision: str, target_directory: Path, files: Sequence[PathLike[str] | str], keep_repo: bool = False
) -> None:
    """
    Clone a specific revision of a Git repository and extract specified files to a target directory.

    Args:
        repo_url: URL of the Git repository to clone.
        revision: Revision of the repository to download.
        target_directory: Directory where the extracted files from the
            repository will be stored.
        files: List of file paths relative to extract relative to the
            root of the Git repository.
        keep_repo: Keeps the cloned repository in the target_directory
            instead of a temporary directory for future re-use
    """

    with contextlib.nullcontext(target_directory) if keep_repo else TemporaryDirectory() as tmpdirname:
        temp_path = Path(tmpdirname) / "temp_repo.git"
        # Only clone the repository if has not already been cloned in keep_repo mode previously
        if keep_repo and temp_path.exists():
            repo = Repo(temp_path)
        else:
            repo = Repo.clone_from(repo_url, temp_path, _GitProgress(), multi_options=["--bare", "--depth 1"])
            # Shallow fetch and checkout the requested revision if necessary
            if repo.head.commit.hexsha != revision:
                result = repo.remote().fetch(revision, progress=_GitProgress(), depth=1)
                assert result[0].commit.hexsha == revision

        stream = BytesIO()
        repo.archive(stream, revision, path=files)
        # Seek back to start to read the in-memory buffer
        stream.seek(0)

        with tarfile.open(fileobj=stream) as tar:
            target_dir = target_directory
            tar.extractall(target_dir)


def _parse_content_disposition_filename(content_disposition: str) -> str | None:
    """
    Parse a Content-Disposition header to extract the filename.

    Args:
        content_disposition: The Content-Disposition header value.

    Returns:
        The extracted filename, or None if no filename is found.
    """

    message = Message()
    message["content-type"] = content_disposition
    filename = message.get_param("filename")
    return filename if isinstance(filename, str) else None


@overload
def download(
    url: str, target_directory: Path | None = None, stream: Literal[True] = True, skip_if_file_exists: bool = False
) -> AbstractContextManager[Iterable[bytes]]: ...


@overload
def download(url: str, target_directory: Path, stream: Literal[False], skip_if_file_exists: bool = False) -> Path: ...


@overload
def download(
    url: str, target_directory: Path | None = None, stream: bool = True, skip_if_file_exists: bool = False
) -> Path | AbstractContextManager[Iterable[bytes]]: ...


def download(
    url: str, target_directory: Path | None = None, stream: bool = True, skip_if_file_exists: bool = False
) -> Path | AbstractContextManager[Iterable[bytes]]:
    """
    Download a file from the given URL.

    Args:
        url: The URL of the file to download.
        target_directory: Optional directory where the file should be
            saved. Ignored when downloading in streaming mode. If not
            provided and streaming is disabled, a `ValueError` will be
            raised.
        stream: Streams the file contents instead of saving it to the
            target directory

    Returns:
        The path of the downloaded file if `target_directory` is
        specified; otherwise, a streaming response.
    """

    # Always enable stream here even when "stream" if False for efficiently saving directly to disk
    response = requests.get(url, stream=True)
    # Enables transfer decompression to address https://github.com/psf/requests/issues/2155
    response.raw.decode_content = True
    file_size = int(response.headers.get("content-length", 0))
    content_disposition = response.headers.get("Content-Disposition")
    response.raise_for_status()

    if content_disposition is None or (filename := _parse_content_disposition_filename(content_disposition)) is None:
        # Parse filename from the URL as a fallback
        filename = Path(urlparse(url).path).name

    with_progress = tqdm.wrapattr(response.raw, "read", total=file_size, desc=filename)
    # Directly return the raw streaming response in streaming mode
    if target_directory is None or stream:
        if not stream:
            raise ValueError("Target directory must be given when not downloading in streaming mode")

        return with_progress  # pyright: ignore

    target_path = target_directory / filename
    # Skip downloading if skipping is enabled and the target file already exists
    if skip_if_file_exists and target_path.is_file():
        return target_path

    # Copy the byte stream directly to a file
    with with_progress as data_stream, target_path.open("wb") as file:
        shutil.copyfileobj(data_stream, file)

    return target_path


def extract_zip(
    path: Iterable[bytes] | Path,
    target_directory: Path,
    members: Iterable[str | Path] | None = None,
    member_globs: bool = False,
) -> list[Path]:
    """
    Extract a zip archive.

    Args:
        path: The file path or bytestream of the zip file.
        target_directory: The directory where the files should be
            extracted.
        members: Specific members to extract. If None, all members are
            extracted.
        member_globs: Whether to interpret the members parameter as
            globs.

    Returns:
        A list of the extracted file paths.
    """

    if not isinstance(path, Path):
        raise NotImplementedError("Streaming decompression not supported for zip files")

    with ZipFile(path) as file:
        if member_globs and members is not None:
            globs = list(map(str, members))
            members = [path for path in map(Path, file.namelist()) if any(path.match(glob) for glob in globs)]

        file.extractall(target_directory, None if members is None else map(str, members))
        if members is None:
            return list(map(Path, file.namelist()))
        return list(map(Path, members))


def extract_tar(
    path_or_stream: Iterable[bytes] | Path,
    target_directory: Path,
    members: Iterable[str | Path] | None = None,
    member_globs: bool = False,
) -> list[Path]:
    """
    Extract a tar archive.

    Args:
        path_or_stream: The file path or bytestream of the tar file.
        target_directory: The directory where the files should be
            extracted.
        members: Specific members to extract. If None, all members are
            extracted.
        member_globs: Whether to interpret the members parameter as
            globs.

    Returns:
        A list of the extracted file paths
    """

    file_argument: dict[Literal["name", "fileobj"], Any]
    if isinstance(path_or_stream, Path):
        file_argument = {"name": path_or_stream}
    else:
        file_argument = {"fileobj": path_or_stream}

    if members is None:
        # Safest default
        filter = "data"
    else:
        if member_globs:
            globs = list(map(str, members))

            # Checks for each path whether they match one of the specified globs
            def tar_filter(tar_info: tarfile.TarInfo, _: str) -> tarfile.TarInfo | None:
                if any(Path(tar_info.name).match(glob) for glob in globs):
                    return tar_info
        else:
            paths = set(map(str, members))

            # Matches paths literally
            def tar_filter(tar_info: tarfile.TarInfo, _: str) -> tarfile.TarInfo | None:
                if tar_info.name in paths:
                    return tar_info

        filter = tar_filter

    with tarfile.open(**file_argument, mode="r") as tar:
        tar.extractall(
            target_directory,
            filter=filter,
        )

    if members is None:
        return [
            path_or_stream
            for directory, _, files in target_directory.walk()
            for file in files
            if (path_or_stream := directory / file).is_file()
        ]

    return list(map(Path, members))


def _get_extractor(
    extension: str,
) -> Callable[[Iterable[bytes] | Path, Path, Iterable[str | Path] | None, bool], list[Path]]:
    """
    Get the appropriate extractor function based on the archive extension.

    Args:
        extension: The archive file extension.

    Returns:
        A callable that extracts files from an archive.
    """

    # Context insensitively find whether the given extension ends in a support archive extension
    extension = extension.lower()
    if any(extension.endswith(supported_extension) for supported_extension in (".tar", ".tar.gz", ".tgz")):
        return extract_tar
    if extension.endswith(".zip"):
        return extract_zip

    raise ValueError(f"Unknown archive extension {extension}")


def extract(
    path_or_stream: Iterable[bytes] | Path,
    target_directory: Path,
    members: Iterable[str | Path] | None = None,
    archive_extension: str | None = None,
    member_globs: bool = False,
) -> list[Path]:
    """
    Extract files from an archive.

    Args:
        path_or_stream: The file path or bytestream of the archive.
        target_directory: The directory where the files should be
            extracted.
        members: Specific members to extract. If None, all members are
            extracted.
        archive_extension: The extension of the archive (optional if
            path_or_stream is a Path).
        member_globs: Whether to interpret the members parameter as
            globs.

    Returns:
        A list of the extracted file paths.
    """

    if isinstance(path_or_stream, Path):
        if archive_extension is None:
            archive_extension = "".join(path_or_stream.suffixes)
    elif archive_extension is None:
        raise ValueError("Archive type needs to be specified when decoding a bytestream")

    extractor = _get_extractor(archive_extension)
    return extractor(path_or_stream, target_directory, members, member_globs)
