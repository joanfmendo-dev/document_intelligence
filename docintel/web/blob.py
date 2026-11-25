"""The master copy of the database in Azure Blob Storage, for servers whose disk is lost at each restart (Container Apps).
At start the files are downloaded; after each write they are uploaded again. To replace the data, upload a new .duckdb to the
container and restart the app. One app instance only: two would overwrite each other."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)


class BlobCopy:
    def __init__(self, container, files: list[Path]):
        """`container` is an azure ContainerClient (or anything with download_blob / upload_blob)."""
        self.container, self.files, self._lock = container, [Path(f) for f in files], threading.Lock()

    @classmethod
    def from_connection_string(cls, connection: str, name: str, files: list[Path]) -> "BlobCopy":
        from azure.storage.blob import ContainerClient  # pip install "docintel[azure]"
        return cls(ContainerClient.from_connection_string(connection, name), files)

    def pull(self) -> None:
        """Download each file that exists in the container (the catalog may not exist yet)."""
        from azure.core.exceptions import ResourceNotFoundError
        for file in self.files:
            try:
                data = self.container.download_blob(file.name).readall()
            except ResourceNotFoundError:
                log.info("%s is not in Blob Storage yet", file.name)
                continue
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(data)

    def push(self) -> None:
        with self._lock:
            for file in self.files:
                if file.is_file():
                    self.container.upload_blob(file.name, file.read_bytes(), overwrite=True)
