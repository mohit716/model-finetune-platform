"""
Storage abstraction layer — supports local filesystem and MinIO (S3-compatible).
Swap backends via STORAGE_BACKEND env var.
"""

import os
import shutil
from abc import ABC, abstractmethod
from typing import Optional
from loguru import logger

from app.config import settings


class StorageBackend(ABC):
    """Abstract storage interface."""

    @abstractmethod
    def save_file(self, source_path: str, dest_key: str) -> str:
        """Save a file, return the storage path/key."""
        ...

    @abstractmethod
    def get_file_path(self, key: str) -> str:
        """Get the local-accessible path for a stored file."""
        ...

    @abstractmethod
    def delete_file(self, key: str) -> bool:
        """Delete a file from storage."""
        ...

    @abstractmethod
    def file_exists(self, key: str) -> bool:
        """Check if a file exists."""
        ...

    @abstractmethod
    def get_file_size(self, key: str) -> int:
        """Get file size in bytes."""
        ...


class LocalStorage(StorageBackend):
    """Local filesystem storage."""

    def __init__(self, base_path: str):
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)

    def _full_path(self, key: str) -> str:
        return os.path.join(self.base_path, key)

    def save_file(self, source_path: str, dest_key: str) -> str:
        dest = self._full_path(dest_key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(source_path, dest)
        logger.info(f"Saved file: {dest_key}")
        return dest_key

    def save_bytes(self, data: bytes, dest_key: str) -> str:
        dest = self._full_path(dest_key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        logger.info(f"Saved bytes: {dest_key} ({len(data)} bytes)")
        return dest_key

    def get_file_path(self, key: str) -> str:
        return self._full_path(key)

    def delete_file(self, key: str) -> bool:
        path = self._full_path(key)
        if os.path.isfile(path):
            os.remove(path)
            return True
        elif os.path.isdir(path):
            shutil.rmtree(path)
            return True
        return False

    def file_exists(self, key: str) -> bool:
        return os.path.exists(self._full_path(key))

    def get_file_size(self, key: str) -> int:
        path = self._full_path(key)
        if os.path.isfile(path):
            return os.path.getsize(path)
        return 0


class MinIOStorage(StorageBackend):
    """MinIO (S3-compatible) storage."""

    def __init__(self):
        from minio import Minio

        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self.bucket = settings.MINIO_BUCKET
        self._ensure_bucket()

        # Local cache for training access
        self.cache_dir = os.path.join(settings.STORAGE_PATH, "minio_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
            logger.info(f"Created MinIO bucket: {self.bucket}")

    def save_file(self, source_path: str, dest_key: str) -> str:
        self.client.fput_object(self.bucket, dest_key, source_path)
        logger.info(f"Uploaded to MinIO: {dest_key}")
        return dest_key

    def get_file_path(self, key: str) -> str:
        """Download from MinIO to local cache and return path."""
        cache_path = os.path.join(self.cache_dir, key)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if not os.path.exists(cache_path):
            self.client.fget_object(self.bucket, key, cache_path)
        return cache_path

    def delete_file(self, key: str) -> bool:
        try:
            self.client.remove_object(self.bucket, key)
            return True
        except Exception:
            return False

    def file_exists(self, key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, key)
            return True
        except Exception:
            return False

    def get_file_size(self, key: str) -> int:
        try:
            stat = self.client.stat_object(self.bucket, key)
            return stat.size
        except Exception:
            return 0


# ── Factory ──────────────────────────────────────────────

def get_storage() -> StorageBackend:
    """Return the configured storage backend."""
    if settings.STORAGE_BACKEND == "minio":
        return MinIOStorage()
    return LocalStorage(settings.STORAGE_PATH)


# Global singleton
storage = get_storage()
