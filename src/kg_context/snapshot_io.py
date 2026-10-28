"""KG-snapshot file access: STREAM the gzipped snapshot, never decompress it to disk.

The released snapshots are `kg_snapshot_<year>-01-01.tsv.gz` (~0.4-0.6 GB each; ~2-2.5 GB each, ~22 GB
total, if decompressed). Every reader goes through `open_kg_snapshot`, which streams the `.gz`
line-by-line. A path may be given with or without the `.gz` suffix (older configs name the plain
`.tsv`): the `.gz` is preferred whenever it exists, and a plain `.tsv` is used only if no `.gz` is
present. Preferring the `.gz` also avoids silently reading a partially-decompressed `.tsv`.
"""
from __future__ import annotations

import gzip
import os
from typing import IO


def resolve_kg_snapshot(path: str) -> str | None:
    """Return the on-disk snapshot file to read for `path` (the `.gz` if present), or None if absent."""
    gz_path = path if path.endswith('.gz') else path + '.gz'
    if os.path.exists(gz_path):
        return gz_path
    if not path.endswith('.gz') and os.path.exists(path):
        return path
    return None


def open_kg_snapshot(path: str) -> IO[str]:
    """Open a KG snapshot for text reading, streaming the `.gz` directly. Raises if absent."""
    resolved = resolve_kg_snapshot(path)
    if resolved is None:
        raise FileNotFoundError(f'KG snapshot not found: {path} (nor {path}.gz)')
    if resolved.endswith('.gz'):
        return gzip.open(resolved, 'rt', encoding='utf-8')
    return open(resolved, 'r', encoding='utf-8')
