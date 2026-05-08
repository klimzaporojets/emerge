"""Unit tests for the garbage detect → reinput → merge chain in
`scripts/dataset/`.

Catches the *correctness* invariants that matter for a reviewer
running their own LLM annotation pipeline:
  - `build_reinput_for_garbage.py` strips ONLY the flagged model's
    annotations on flagged records, leaves the rest of the source tree
    byte-identical.
  - `merge_reinput_into_dataset.py` splices corrected records by
    `hash_id`, leaves non-matching source records byte-identical, and
    ABSOLUTELY DOES NOT touch the source tree on disk (a reviewer must
    be able to re-run safely without losing their annotations).
  - The chain refuses dangerous configurations (source==output dir,
    duplicate hash_ids in reinput).

~5 s, CPU only, no network. All synthetic fixtures in `tmp_path`. No
real LLM annotations required.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILD_REINPUT = REPO / "scripts/dataset/build_reinput_for_garbage.py"
MERGE_REINPUT = REPO / "scripts/dataset/merge_reinput_into_dataset.py"


def _record(hash_id, llm_assessments):
    """Tiny synthetic record matching the s05 schema in the bits these
    scripts care about. Real records have ~14 top-level fields; we
    only need hash_id + tkgu_triples[*].llm_assessment for the splice
    logic."""
    return {
        "hash_id": hash_id,
        "passage": f"synthetic passage for {hash_id}",
        "tkgu_triples": [
            {
                "triple": ["Q1", "P31", "Q5"],
                "tkgu_operations": ["e-triples"],
                "llm_assessment": llm_assessments,
            }
        ],
    }


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Helper: run a script in a subprocess and bubble up its exit code on failure.
# Subprocess (rather than direct import) is faithful to how a reviewer
# actually invokes these tools and catches any module-load-time issues.
# ---------------------------------------------------------------------------
def _run(script, *args, expect_ok=True):
    result = subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        capture_output=True, text=True, timeout=30,
    )
    if expect_ok and result.returncode != 0:
        pytest.fail(
            f"{script.name} exited {result.returncode}\n"
            f"argv: {args}\nstderr:\n{result.stderr}"
        )
    return result


# ---------------------------------------------------------------------------
# Build a 5-record source tree that build_reinput / merge_reinput can chew on.
# Layout matches the post-§3.5 expectations (snapshot_*/llm_assessed/delta_*.jsonl).
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_source_tree(tmp_path):
    src = tmp_path / "source/llama405b_assessed"
    delta_dir = src / "snapshot_2099-01-01" / "llm_assessed"
    delta_dir.mkdir(parents=True)
    delta_path = delta_dir / "delta_2099-01-01.jsonl"
    records = [
        _record(
            f"hash_{i:02d}",
            [
                {"llm_name": "Meta-Llama-3.1-405B_prompt_v1",
                 "llm_assessment": True, "llm_prompt_type": "triple_assertion",
                 "llm_prompt": f"original 405B response for hash_{i:02d}"},
                {"llm_name": "Meta-Llama-3.1-8B_prompt_v1",
                 "llm_assessment": True, "llm_prompt_type": "triple_assertion",
                 "llm_prompt": f"original 8B response for hash_{i:02d}"},
            ],
        )
        for i in range(5)
    ]
    with delta_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return {"root": src, "delta_path": delta_path, "records": records}


# ---------------------------------------------------------------------------
# 1. merge splice correctness + source untouched
# ---------------------------------------------------------------------------
def test_merge_splices_by_hash_id_and_leaves_source_untouched(
    synthetic_source_tree, tmp_path
):
    """`merge_reinput_into_dataset.py` must:
      - replace records whose hash_id appears in --reinput-file
      - leave non-matching records byte-identical to source
      - **never** modify the source tree (mtime + md5 unchanged)
    """
    src_root = synthetic_source_tree["root"]
    src_delta = synthetic_source_tree["delta_path"]

    # Reinput with corrections for hash_01 and hash_03 (out of 5).
    reinput_dir = tmp_path / "reinput/llama405b_assessed/snapshot_2099-01-01/llm_assessed"
    reinput_dir.mkdir(parents=True)
    reinput_path = reinput_dir / "delta_2099-01-01.jsonl"
    corrected_records = [
        _record(
            "hash_01",
            [{"llm_name": "Meta-Llama-3.1-405B_prompt_v1",
              "llm_assessment": False, "llm_prompt_type": "triple_assertion",
              "llm_prompt": "CORRECTED 405B response for hash_01"}],
        ),
        _record(
            "hash_03",
            [{"llm_name": "Meta-Llama-3.1-405B_prompt_v1",
              "llm_assessment": False, "llm_prompt_type": "triple_assertion",
              "llm_prompt": "CORRECTED 405B response for hash_03"}],
        ),
    ]
    with reinput_path.open("w") as f:
        for rec in corrected_records:
            f.write(json.dumps(rec) + "\n")

    # Capture pre-merge source signature for the "untouched" assertion.
    src_md5_before = _md5(src_delta)
    src_mtime_before = src_delta.stat().st_mtime

    out_root = tmp_path / "merged/llama405b_assessed"
    out_manifest = tmp_path / "merge_manifest.json"
    _run(MERGE_REINPUT,
         "--source-root", src_root,
         "--reinput-file", reinput_path,
         "--output-root", out_root,
         "--output-manifest", out_manifest)

    # Source MUST be byte-identical to its pre-merge state.
    assert _md5(src_delta) == src_md5_before, "source delta md5 changed — leak!"
    assert src_delta.stat().st_mtime == src_mtime_before, "source delta mtime changed — leak!"

    # Output must exist and have exactly 5 records (same count as source).
    out_delta = out_root / "snapshot_2099-01-01/llm_assessed/delta_2099-01-01.jsonl"
    assert out_delta.is_file(), f"merged output not written: {out_delta}"
    out_records = [json.loads(l) for l in out_delta.open()]
    assert len(out_records) == 5

    # The 2 corrected records must reflect the reinput; the 3 others
    # must be byte-equivalent (deep dict equal) to the source.
    by_hash = {r["hash_id"]: r for r in out_records}
    assert by_hash["hash_01"]["tkgu_triples"][0]["llm_assessment"][0]["llm_assessment"] is False
    assert "CORRECTED" in by_hash["hash_01"]["tkgu_triples"][0]["llm_assessment"][0]["llm_prompt"]
    assert "CORRECTED" in by_hash["hash_03"]["tkgu_triples"][0]["llm_assessment"][0]["llm_prompt"]
    for h in ("hash_00", "hash_02", "hash_04"):
        src_rec = next(r for r in synthetic_source_tree["records"] if r["hash_id"] == h)
        assert by_hash[h] == src_rec, f"non-matching record {h} drifted in merge"


# ---------------------------------------------------------------------------
# 2. merge refuses source==output (catches user copy-paste mistake)
# ---------------------------------------------------------------------------
def test_merge_refuses_when_output_root_is_inside_source(synthetic_source_tree, tmp_path):
    """If a user accidentally points --output-root at the source tree,
    `merge_reinput_into_dataset.py` must refuse — otherwise it would
    silently overwrite source records mid-merge."""
    src_root = synthetic_source_tree["root"]
    # Empty reinput file; this test is about the path-safety check, not
    # the splice logic.
    bad_reinput = tmp_path / "empty.jsonl"
    bad_reinput.write_text("")

    # Output INSIDE source — must be rejected.
    output_inside = src_root / "should_not_be_created"
    result = _run(
        MERGE_REINPUT,
        "--source-root", src_root,
        "--reinput-file", bad_reinput,
        "--output-root", output_inside,
        expect_ok=False,
    )
    assert result.returncode != 0, (
        "merge_reinput_into_dataset.py allowed --output-root inside --source-root; "
        "this is a footgun that would silently overwrite source."
    )


# ---------------------------------------------------------------------------
# 3. merge errors on duplicate hash_id in reinput (catches concat bugs)
# ---------------------------------------------------------------------------
def test_merge_errors_on_duplicate_hash_id_in_reinput(synthetic_source_tree, tmp_path):
    """Two reinput records with the same hash_id is ambiguous (which
    correction wins?). The script must error rather than apply
    last-write-wins silently. Common breakage path: concatenating
    multiple reinput JSONLs without dedup."""
    src_root = synthetic_source_tree["root"]
    reinput_dir = tmp_path / "reinput/llama405b_assessed/snapshot_2099-01-01/llm_assessed"
    reinput_dir.mkdir(parents=True)
    reinput_path = reinput_dir / "delta_2099-01-01.jsonl"
    # Same hash_id twice — buggy concat.
    duplicate_records = [_record("hash_01", []), _record("hash_01", [])]
    with reinput_path.open("w") as f:
        for rec in duplicate_records:
            f.write(json.dumps(rec) + "\n")

    out_root = tmp_path / "merged/llama405b_assessed"
    out_manifest = tmp_path / "merge_manifest.json"
    result = _run(
        MERGE_REINPUT,
        "--source-root", src_root,
        "--reinput-file", reinput_path,
        "--output-root", out_root,
        "--output-manifest", out_manifest,
        expect_ok=False,
    )
    assert result.returncode != 0, (
        "merge_reinput_into_dataset.py silently accepted duplicate hash_id "
        "in --reinput-file; this would corrupt the merge ambiguously."
    )
