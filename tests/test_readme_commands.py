"""Every script / module / config path a README tells a user to run must EXIST.

Guards the #1 public-repo failure: a user copy-pastes a command from a .md and it
fails because the script was renamed, a `python -m` module moved, or a
`--config_file` path is stale (exactly the s06b / s0x / _vN rename classes we hit).

Pure static check (no imports, no network, no data) — walks the working tree, so it
also sees not-yet-committed renames. Parametrized so a failure names the offending
.md line and the missing target.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Directories that are NOT part of the source tree (runtime data, VCS, private, caches).
SKIP_DIRS = {".git", "_private", "data", "output", "logs", "__pycache__", ".pytest_cache"}

# Tokens that are deliberately placeholders (user fills them in) — never real paths.
PLACEHOLDER_MARKERS = (
    "/path/to", "path/to/your", "<", ">", "{year}", "${", "yyyy", "YYYY",
    "<snap", "<year", "<config", "edc-plus-config", "your_", "...",
)
# LEADING path prefixes of build artifacts / runtime data that are legitimately absent.
# Matched as a prefix only (see _is_absent_ok): a bare substring test made "data/" match
# "wikidata/" and silently skipped 14 real scripts/slurm/wikidata + config/dataset/wikidata refs.
ABSENT_OK = (
    "src/dataset/wikidata/target/",   # Maven build output
    "data/",                          # populated by download_data.sh
    "output/",                        # produced at runtime
)


def _is_absent_ok(token: str) -> bool:
    """True iff `token` lives under a runtime/build-artifact root (leading-prefix match only)."""
    t = token[2:] if token.startswith("./") else token
    return any(t.startswith(a) for a in ABSENT_OK)

PY_MODULE_RE = re.compile(r"-m\s+([A-Za-z_][A-Za-z0-9_.]+)")
SCRIPT_RE = re.compile(r"(?<![\w./])\.?/?((?:scripts|src)/[A-Za-z0-9_./-]+\.(?:sh|py))")
CONFIG_RE = re.compile(r"(?<![\w./])(config/[A-Za-z0-9_./-]+\.json)")


def _md_files():
    out = []
    for p in REPO.rglob("*.md"):
        if any(part in SKIP_DIRS for part in p.relative_to(REPO).parts):
            continue
        out.append(p)
    return sorted(out)


def _is_placeholder(tok: str) -> bool:
    low = tok.lower()
    return any(m.lower() in low for m in PLACEHOLDER_MARKERS)


def _collect():
    """Yield (md_relpath, lineno, kind, token, target_path) for every executable reference."""
    refs = []
    for md in _md_files():
        rel = str(md.relative_to(REPO))
        for i, line in enumerate(md.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for m in PY_MODULE_RE.findall(line):
                if _is_placeholder(m):
                    continue
                refs.append((rel, i, "module", m, REPO / "src" / (m.replace(".", "/") + ".py")))
            for s in SCRIPT_RE.findall(line):
                if _is_placeholder(s):
                    continue
                refs.append((rel, i, "script", s, REPO / s))
            for c in CONFIG_RE.findall(line):
                if _is_placeholder(c) or "*" in c:
                    continue
                refs.append((rel, i, "config", c, REPO / c))
    return refs


_REFS = _collect()


def test_found_some_references():
    # Guards against the regexes silently matching nothing (e.g. after a refactor).
    assert len(_REFS) > 10, f"suspiciously few README references found ({len(_REFS)})"


@pytest.mark.parametrize("token,expected", [
    ("data/evaluation_set/x.py", True),                  # download dir -> legitimately absent
    ("./output/s04/run.py", True),                       # runtime output
    ("src/dataset/wikidata/target/app.py", True),        # Maven build output
    ("scripts/slurm/wikidata/s03_get_kg_snapshot.sh", False),   # 'wikidata/' must NOT match 'data/'
    ("config/dataset/wikidata/s03_get_kg_snapshot/20250320/config.json", False),
    ("src/dataset/wikidata/python/s03_get_kg_snapshot.py", False),
    ("scripts/metadata/foo.py", False),                  # any '*data/' dir is not the data/ root
])
def test_absent_ok_is_prefix_not_substring(token, expected):
    """Regression guard: the runtime-artifact allow-list must not swallow real source paths."""
    assert _is_absent_ok(token) is expected


@pytest.mark.parametrize(
    "md,lineno,kind,token,target",
    _REFS,
    ids=[f"{r[0]}:{r[1]}:{r[3]}" for r in _REFS],
)
def test_readme_reference_exists(md, lineno, kind, token, target):
    if _is_absent_ok(token):
        pytest.skip(f"{token} is a runtime/build artifact (legitimately absent)")
    assert target.exists(), (
        f"{md}:{lineno} references a {kind} that does not exist on disk:\n"
        f"    {token}\n"
        f"    expected at: {target.relative_to(REPO)}\n"
        f"If it was renamed, update the README (or the file); if it's a placeholder, "
        f"make it obviously one (e.g. <snapshot>, /path/to/...)."
    )
