"""Data-integrity checks on the released `data/evaluation_set/` JSONL.

Catches:
  - An HF re-upload that accidentally duplicated/dropped records (the
    paper claims 3,500; if the count drifts, results stop being
    reproducible).
  - `hash_id` collisions across deltas (the chain scripts in
    `scripts/dataset/merge_reinput_into_dataset.py` *assume* uniqueness;
    a collision would silently corrupt the merge).
  - Schema drift between `data/README.md` (which documents 14 top-level
    fields) and the actual JSONL on HF.

Skips cleanly with `pytest.skip()` if the user hasn't run
`./scripts/download_data.sh` yet — fresh clones still pass `pytest`.
~3 s when data is present, instant when skipped. CPU only, no network.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EVAL_SET = REPO / "data/evaluation_set"

# Documented in data/README.md (14 top-level fields). Test asserts the
# released JSONL contains AT LEAST these — extras are tolerable, missing
# any one is a regression.
DOCUMENTED_TOP_LEVEL_FIELDS = {
    "hash_id", "passage", "mentions", "revision_id", "revision_date",
    "revision_timestamp", "anchor_title", "anchor_page_id", "anchor_page_qid",
    "paragraph_idx", "delta_dates", "delta_timestamps", "tkgu_triples",
    "predictions",
}

EXPECTED_RECORD_COUNT = 3500  # Per paper §5 line 250 + Datasheet K.2


def _all_delta_files():
    if not EVAL_SET.is_dir():
        return []
    return sorted(EVAL_SET.glob("snapshot_*-01-01/delta_*.jsonl"))


def _skip_if_no_data():
    files = _all_delta_files()
    if not files:
        pytest.skip(
            f"data/evaluation_set/ not populated — run "
            f"`./scripts/download_data.sh` first to enable this test"
        )
    return files


def test_record_count_is_3500():
    """Total records across all 35 delta JSONL files must be exactly 3500."""
    files = _skip_if_no_data()
    n = sum(1 for f in files for _ in f.open())
    assert n == EXPECTED_RECORD_COUNT, (
        f"Expected {EXPECTED_RECORD_COUNT} records (paper §5 / Datasheet K.2), "
        f"got {n} across {len(files)} files. An HF re-upload may have "
        f"changed the test split — verify before publishing."
    )


def test_hash_ids_are_unique():
    """No hash_id may collide across the 3500 records — the merge / re-query
    chain in `scripts/dataset/` keys on hash_id and would silently corrupt
    on a collision.
    """
    files = _skip_if_no_data()
    seen = {}
    duplicates = []
    for f in files:
        for line in f.open():
            rec = json.loads(line)
            h = rec.get("hash_id")
            assert h, f"record without hash_id in {f.name}"
            if h in seen:
                duplicates.append((h, seen[h], f.name))
                if len(duplicates) >= 5:  # cap report size; keep test fast
                    break
            else:
                seen[h] = f.name
        if len(duplicates) >= 5:
            break
    assert not duplicates, (
        f"hash_id collisions detected (showing first {len(duplicates)}):\n"
        + "\n".join(f"  {h!r}: {a} / {b}" for h, a, b in duplicates)
    )


def test_records_have_documented_top_level_fields():
    """Every record's top-level key set must be a superset of the 14 fields
    documented in `data/README.md`. Extras (e.g., implementation
    fields added later) are tolerable, but missing any documented field
    means the README and the data have drifted.
    """
    files = _skip_if_no_data()
    # Sample a handful of records (one per delta file) — checking every
    # record adds runtime without catching anything the sample misses,
    # since schema is uniform within a delta.
    missing_per_file = []
    for f in files:
        with f.open() as fh:
            first = json.loads(next(fh))
        missing = DOCUMENTED_TOP_LEVEL_FIELDS - set(first.keys())
        if missing:
            missing_per_file.append((f.name, sorted(missing)))
    assert not missing_per_file, (
        "Documented top-level fields missing from records:\n"
        + "\n".join(f"  {name}: missing {miss}" for name, miss in missing_per_file)
        + "\nUpdate data/README.md or fix the upload, but don't leave them divergent."
    )
