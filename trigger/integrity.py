"""Bind private diagnostic results to exact immutable input-file bytes."""
from __future__ import annotations

import hashlib
from pathlib import Path


def digest(path):
    path = Path(path)
    sha = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(8 * 1024 * 1024):
            sha.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha.hexdigest()}


def files(paths):
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    return [digest(path) for path in paths]
