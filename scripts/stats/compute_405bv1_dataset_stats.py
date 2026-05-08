#!/usr/bin/env python3
"""Compute per-operation and per-snapshot triple counts on the merged_v2
corpus, applied under the rule-C (405B-prompt-v1) filter.

Mirrors the logic of `scripts/stats/verify_paper_complete_dataset_filter.py`
(which uses the rule-A / 8B-supported filter on s07_reduce_duplicates),
but runs on the merged_v2 corpus tree under the rule-C filter:

  Rule C — assertion ops:
      verdict(triple, "Meta-Llama-3.1-405B_prompt_v1", "triple_assertion") == True
  Rule C — deprecation ops:
      verdict(triple, "Meta-Llama-3.1-405B_prompt_v1", "triple_deprecation") == True

Outputs:
  - Console: per-snapshot kept-instance/kept-ops counts; per-op kept counts;
    headline `\\statsTotnrtkgus`-equivalent under 405B-v1.
  - --output-tex: a LaTeX macros file with `\\newcommand{\\statsTotnrtkgusVone}{N}`
    + per-op `\\statsXTriplesVone` etc., ready to copy into the manuscript.
  - --output-json: machine-readable summary for downstream scripts.

Usage (typical: against the public HF release, after `./scripts/download_data.sh --corpus`):

    python scripts/stats/compute_405bv1_dataset_stats.py \\
        --root data/corpus \\
        --output-tex output/stats_405bv1_macros.tex \\
        --output-json output/stats_405bv1.json

The --root may point at either the flat HF layout (snapshot_*/delta_*.jsonl,
auto-detected) or an internal nested merged-tree layout
(snapshot_*/llm_assessed/delta_*.jsonl).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


LLM_405B_V1 = "Meta-Llama-3.1-405B_prompt_v1"
PROMPT_ASSERT = "triple_assertion"
PROMPT_DEP = "triple_deprecation"

YEARLY = [
    "snapshot_2019-01-01", "snapshot_2020-01-01", "snapshot_2021-01-01",
    "snapshot_2022-01-01", "snapshot_2023-01-01", "snapshot_2024-01-01",
    "snapshot_2025-01-01",
]


def verdict(triple, llm_name, prompt_type):
    """True iff llm_assessment has an entry matching (llm_name, prompt_type)
    with value True. Missing entries → False."""
    for a in triple.get("llm_assessment", []):
        if a.get("llm_name") == llm_name and a.get("llm_prompt_type") == prompt_type:
            return a.get("llm_assessment") is True
    return False


def keep_triple_rule_c(triple):
    """Rule C: 405B-v1 verdict (assertion or deprecation depending on op)."""
    ops = set(triple.get("tkgu_operations") or [])
    if "d-triples" in ops:
        return verdict(triple, LLM_405B_V1, PROMPT_DEP)
    return verdict(triple, LLM_405B_V1, PROMPT_ASSERT)


def primary_op(triple):
    ops = triple.get("tkgu_operations") or ["UNKNOWN"]
    return ops[0]


def macroize(name):
    """Convert TKGU op label like 'ee-kg-triples' to a LaTeX-macro-safe stem."""
    mapping = {
        "x-triples": "XTriplesVone",
        "e-triples": "ETriplesVone",
        "ee-triples": "EEtriplesVone",
        "ee-kg-triples": "EEKGtriplesVone",
        "d-triples": "DtriplesVone",
    }
    return mapping.get(name, name.replace("-", "").capitalize() + "Vone")


def fmt_macroval(n):
    """Format integer with LaTeX thousands separators: 1234567 -> 1{,}234{,}567"""
    return "{,}".join(f"{n:,}".split(","))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", type=Path, required=True,
                    help="Path to merged-v2 dataset tree (`llama405b_assessed/`).")
    ap.add_argument("--output-tex", type=Path, required=True,
                    help="Output .tex file with LaTeX macros for paste into manuscript.")
    ap.add_argument("--output-json", type=Path, required=True,
                    help="Output .json file with machine-readable summary.")
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f"ERROR: --root does not exist: {args.root}")

    snap_in_inst = Counter()
    snap_kept_inst = Counter()
    snap_kept_ops = Counter()
    op_kept = Counter()
    op_total = Counter()
    n_in = 0
    n_kept_inst = 0
    n_records_with_at_least_one_v1 = 0

    print(f"[stats_405bv1] scanning {args.root} ...", flush=True)
    for snap in YEARLY:
        # Auto-detect layout: nested (snapshot_*/llm_assessed/delta_*.jsonl, the
        # internal merged-tree layout) or flat (snapshot_*/delta_*.jsonl, what
        # the public HF release / `download_data.sh --corpus` produces).
        snap_dir_nested = args.root / snap / "llm_assessed"
        snap_dir_flat = args.root / snap
        if snap_dir_nested.is_dir():
            snap_dir = snap_dir_nested
        elif snap_dir_flat.is_dir():
            snap_dir = snap_dir_flat
        else:
            print(f"  WARN: missing {snap_dir_nested} (also tried {snap_dir_flat})", flush=True)
            continue
        for delta_path in sorted(snap_dir.glob("delta_*.jsonl")):
            with delta_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    n_in += 1
                    snap_in_inst[snap] += 1
                    has_v1 = False
                    kept_triples_in_record = 0
                    for t in rec.get("tkgu_triples", []):
                        op = primary_op(t)
                        op_total[op] += 1
                        # Track 405B-v1 presence (any value, any prompt_type).
                        if not has_v1:
                            for a in t.get("llm_assessment", []):
                                if a.get("llm_name") == LLM_405B_V1:
                                    has_v1 = True
                                    break
                        if keep_triple_rule_c(t):
                            kept_triples_in_record += 1
                            op_kept[op] += 1
                            snap_kept_ops[snap] += 1
                    if has_v1:
                        n_records_with_at_least_one_v1 += 1
                    if kept_triples_in_record > 0:
                        n_kept_inst += 1
                        snap_kept_inst[snap] += 1

    total_kept_ops = sum(op_kept.values())
    total_ops = sum(op_total.values())

    print()
    print("=" * 70)
    print("=== Per-snapshot ===")
    print("=" * 70)
    print(f"  {'Snapshot':<25} {'Instances':>10} {'Ops kept (v1)':>15}")
    for snap in YEARLY:
        print(f"  {snap:<25} {snap_in_inst[snap]:>10,} {snap_kept_ops[snap]:>15,}")
    print(f"  {'TOTAL':<25} {n_in:>10,} {total_kept_ops:>15,}")

    print()
    print("=" * 70)
    print("=== Per-op kept (under rule-C / 405B-v1 filter) ===")
    print("=" * 70)
    for op in sorted(op_total.keys()):
        kept = op_kept[op]
        total = op_total[op]
        keep_rate = 100 * kept / max(total, 1)
        print(f"  {op:<20} kept={kept:>10,} / total={total:>10,}  ({keep_rate:.1f}% kept)")

    print()
    print("=" * 70)
    print("=== Headline numbers ===")
    print("=" * 70)
    print(f"  records (instances):                        {n_in:,}")
    print(f"  records with ≥1 405B-v1 verdict:            {n_records_with_at_least_one_v1:,} ({100*n_records_with_at_least_one_v1/max(n_in,1):.2f}%)")
    print(f"  raw tkgu_triples:                           {total_ops:,}")
    print(f"  tkgu_triples kept under 405B-v1 (rule-C):   {total_kept_ops:,} ({100*total_kept_ops/max(total_ops,1):.2f}% kept)")
    print(f"  rule-C \\statsTotnrtkgusVone equivalent:     {total_kept_ops:,}")

    # Write JSON
    summary = {
        "root": str(args.root.resolve()),
        "filter": "rule-C (405B-prompt-v1)",
        "n_records": n_in,
        "n_records_with_v1": n_records_with_at_least_one_v1,
        "n_raw_tkgu_triples": total_ops,
        "n_kept_tkgu_triples": total_kept_ops,
        "per_op_kept": dict(op_kept),
        "per_op_total": dict(op_total),
        "per_snap_records": {s: snap_in_inst[s] for s in YEARLY},
        "per_snap_kept_ops": {s: snap_kept_ops[s] for s in YEARLY},
        "per_snap_kept_records": {s: snap_kept_inst[s] for s in YEARLY},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[stats_405bv1] JSON written: {args.output_json}")

    # Write LaTeX macros
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tex.open("w") as fh:
        fh.write("% LaTeX macros — dataset statistics under rule-C (405B-prompt-v1) filter.\n")
        fh.write(f"% Generated from: {args.root.resolve()}\n")
        fh.write("% Suffix `Vone` denotes the 405B-prompt-v1 filter to distinguish from\n")
        fh.write("% existing 8B-filter macros.\n\n")
        fh.write(f"\\newcommand{{\\statsNuminstancesVone}}{{{fmt_macroval(n_in)}}}\n")
        fh.write(f"\\newcommand{{\\statsTotnrtkgusVone}}{{{fmt_macroval(total_kept_ops)}}}\n\n")
        for op in sorted(op_total.keys()):
            kept = op_kept[op]
            macro = macroize(op)
            fh.write(f"\\newcommand{{\\stats{macro}}}{{{fmt_macroval(kept)}}}  % {op}\n")
        fh.write("\n% Per-snapshot kept ops (for any per-year tables):\n")
        for snap in YEARLY:
            year = snap.split("_")[1].split("-")[0]
            fh.write(f"\\newcommand{{\\statsKeptOps{year}Vone}}{{{fmt_macroval(snap_kept_ops[snap])}}}\n")
    print(f"[stats_405bv1] TeX macros written: {args.output_tex}")


if __name__ == "__main__":
    main()
