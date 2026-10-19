"""No author / cluster / private-workspace fingerprints in the public tree.

Guards the "leaked my laptop path / Snellius / private repo name" class: an
absolute author path, an HPC mount, a cluster name, or the private staging repo's
name slipping into a tracked file. Complements tests/test_config_files.py (which
only scans config/) by scanning the WHOLE source tree.

Static, offline, zero-config. Walks the working tree (so it also sees not-yet-
committed files). For notebooks it parses JSON and scans only cell *source*, so
base64 image outputs can't produce false positives.
"""
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SELF = Path(__file__).name

SKIP_DIRS = {".git", ".claude", "_private", "data", "output", "logs", "__pycache__", ".pytest_cache"}
TEXT_EXT = {".py", ".sh", ".json", ".md", ".txt", ".yaml", ".yml", ".cfg", ".toml", ".ipynb"}

# Specific author / cluster / private fingerprints. Deliberately narrow so they do
# NOT match the generic leak-guard regexes in test_config_files.py, nor legitimate
# placeholders like /path/to/... or ~/.  (case-insensitive)
PATTERNS = [
    r"/home/klim",          # author home dir
    r"/projects/0/",        # Snellius project mount
    r"/scratch-shared",     # Snellius scratch
    r"\bprjs1103\b",        # author's HPC project allocation
    r"\bkzaporojets\b",     # author username
    r"\bsnellius\b",        # the HPC cluster by name
    r"emerge-stage",        # the PRIVATE paper repo
]
RE = re.compile("|".join(PATTERNS), re.IGNORECASE)


def _iter_text_files():
    for p in REPO.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(REPO)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.name == SELF:                      # this file defines the patterns
            continue
        if p.suffix.lower() in TEXT_EXT:
            yield p, rel


def _lines_to_scan(path: Path):
    """For .ipynb, only cell source; for everything else, all lines."""
    if path.suffix.lower() == ".ipynb":
        try:
            nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return
        for cell in nb.get("cells", []) if isinstance(nb, dict) else []:
            src = cell.get("source", []) if isinstance(cell, dict) else []
            if isinstance(src, str):
                src = src.splitlines()
            for n, line in enumerate(src, 1):
                yield n, line
    else:
        try:
            for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                yield n, line
        except Exception:
            return


def _collect_offenders():
    offenders = []
    for path, rel in _iter_text_files():
        for n, line in _lines_to_scan(path):
            m = RE.search(line)
            if m:
                offenders.append(f"  {rel}:{n} -> {m.group(0)!r}")
    return offenders


def test_no_author_or_cluster_paths():
    offenders = _collect_offenders()
    assert not offenders, (
        "Author / cluster / private fingerprints found in tracked files "
        "(use /path/to/... placeholders, ~, or generic wording):\n" + "\n".join(offenders)
    )
