"""Validity + path hygiene of all JSON configs under config/.

Catches:
  - Malformed JSON anywhere in `config/` (a single typo in a config file
    that the runtime only opens lazily can sit unnoticed for weeks).
  - Snellius / author-side paths leaking back into the *ported*
    405B-v1 configs after a future refactor (the migration scrubbed
    them once; this test ensures it stays scrubbed).

<1 s, CPU only, no data needed. Failure messages name the offending file
and line so debugging is one click away.
"""
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Discover every JSON config at import time, sort for deterministic test IDs.
ALL_CONFIGS = sorted((REPO / "config").rglob("*.json"))


@pytest.mark.parametrize(
    "config_path",
    ALL_CONFIGS,
    ids=lambda p: str(p.relative_to(REPO)),
)
def test_config_is_valid_json(config_path):
    """Every config/**/*.json must parse without error."""
    try:
        with config_path.open() as f:
            json.load(f)
    except json.JSONDecodeError as e:
        pytest.fail(
            f"{config_path.relative_to(REPO)} is invalid JSON: "
            f"{e.msg} at line {e.lineno} col {e.colno}"
        )


# Snellius / author-side path patterns that should never appear in any
# emerge-prod config body. Excludes the documented `/path/to/storage/`
# placeholder which IS the convention.
SNELLIUS_PATTERNS = re.compile(
    r"/projects/0/|/scratch-shared/|prjs1103|gcn[0-9]+|kzaporojets|/home/klim/"
)


def test_no_snellius_paths_in_ported_405bv1_configs():
    """The 8 ported s05 405B-v1 configs must use the `/path/to/storage/`
    placeholder convention, not author-side absolute paths. Was a real
    issue in the original migration; this test prevents accidental
    re-introduction by a copy-paste from emerge-stage.
    """
    ported_dir = (
        REPO
        / "config/dataset/emerge/s05_generate_dataset_with_llm"
        / "20260502_v8_llama_405b_v1_complete_dataset"
    )
    assert ported_dir.is_dir(), f"expected ported config dir at {ported_dir}"
    offenders = []
    for config_path in sorted(ported_dir.glob("*.json")):
        text = config_path.read_text()
        for match in SNELLIUS_PATTERNS.finditer(text):
            line = text[: match.start()].count("\n") + 1
            offenders.append(
                f"  {config_path.relative_to(REPO)}:{line} → {match.group(0)!r}"
            )
    if offenders:
        pytest.fail(
            "Snellius / author-side path residue in ported 405B-v1 configs:\n"
            + "\n".join(offenders)
            + "\nReplace with `/path/to/storage/wikipedia-processing/output/...` "
            "to match the existing emerge-prod config convention."
        )
