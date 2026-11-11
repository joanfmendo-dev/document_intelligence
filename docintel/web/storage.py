"""Uploaded files. An upload only lives while its fields are being reviewed: uploads/<job_id>/<file>.
Once the record is saved (or the upload cancelled) the file is deleted; the original document is never kept."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import BinaryIO

from .._images import IMAGE_SUFFIXES

DOCUMENT_SUFFIXES = {".pdf", ".docx", ".txt"} | IMAGE_SUFFIXES
_IN_BROWSER_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}  # TIFF is not shown by browsers


class FileTooLarge(Exception):
    pass


def safe_name(filename: str) -> str:
    """The file name without any folder, and only with characters that are safe on every file system."""
    name = re.sub(r"[^\w.\- ]+", "_", Path(filename.replace("\\", "/")).name, flags=re.UNICODE).strip(" .")
    return name or "document"


def kind_of(path: Path) -> str:
    """How the viewer shows a file: 'pdf', 'image', 'text', or 'other' (only offered as a download)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in _IN_BROWSER_IMAGES:
        return "image"
    return "text" if suffix == ".txt" else "other"


class UploadStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def prepare(self) -> None:
        """Create the folder and drop the files of reviews that never finished (their jobs lived in memory)."""
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, job_id: str, filename: str, stream: BinaryIO, max_bytes: int) -> Path:
        """Write an uploaded file under a new job folder. Raises FileTooLarge (and leaves nothing behind)."""
        folder = self.root / job_id
        folder.mkdir(parents=True)
        path = folder / safe_name(filename)
        size = 0
        try:
            with open(path, "wb") as out:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise FileTooLarge
                    out.write(chunk)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        if size == 0:
            shutil.rmtree(folder, ignore_errors=True)
            raise ValueError("empty file")
        return path

    def discard(self, path: Path) -> None:
        """Delete an upload (its whole job folder)."""
        if path.parent.parent == self.root:
            shutil.rmtree(path.parent, ignore_errors=True)
