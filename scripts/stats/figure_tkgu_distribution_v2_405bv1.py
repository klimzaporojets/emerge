#!/usr/bin/env python3
"""Regenerate the main-paper TKGU operations distribution pie chart
under the 405B-prompt-v1 (rule-C) filter, replacing the legacy 8B-filter
version produced by `src/stats/dataset/figure_tkgu_distribution.ipynb`.

Output: PDF + PNG matching the original layout (1x2 pies: complete
dataset on the left, subsampled test set on the right).

Reads:
  --merged-root    : path to merged_v2 reordered tree (left pie, complete dataset).
  --eval-set-root  : path to evaluation_set tree (right pie, 3,500-instance test set).

Both pies use the rule-C filter:
  - assertion ops: verdict(triple, "Meta-Llama-3.1-405B_prompt_v1", "triple_assertion") == True
  - deprecation ops: verdict(triple, "Meta-Llama-3.1-405B_prompt_v1", "triple_deprecation") == True

Usage (typical: against the public HF release, after `./scripts/download_data.sh --corpus`):

    python scripts/stats/figure_tkgu_distribution_v2_405bv1.py \\
        --merged-root data/corpus \\
        --eval-set-root data/evaluation_set \\
        --output-pdf output/figures/pie_chart_nr_triples_both_405bv1.pdf \\
        --output-png output/figures/pie_chart_nr_triples_both_405bv1.png

Both --merged-root and --eval-set-root accept either the flat HF layout
(snapshot_*/delta_*.jsonl, auto-detected) or an internal nested
merged-tree layout (snapshot_*/llm_assessed/delta_*.jsonl).
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


LLM_405B_V1 = "Meta-Llama-3.1-405B_prompt_v1"
PROMPT_ASSERT = "triple_assertion"
PROMPT_DEP = "triple_deprecation"

LABEL_MAP = {
    "x-triples": "Exists",
    "e-triples": "Add",
    "ee-triples": "Mint+Add",
    "ee-kg-triples": "Infer",
    "d-triples": "Deprecate",
}
TKGU_ORDER = ["x-triples", "e-triples", "ee-triples", "ee-kg-triples", "d-triples"]


def verdict(triple, llm_name, prompt_type):
    for a in triple.get("llm_assessment", []):
        if a.get("llm_name") == llm_name and a.get("llm_prompt_type") == prompt_type:
            return a.get("llm_assessment") is True
    return False


def keep_triple_rule_c(triple):
    ops = set(triple.get("tkgu_operations") or [])
    if "d-triples" in ops:
        return verdict(triple, LLM_405B_V1, PROMPT_DEP)
    return verdict(triple, LLM_405B_V1, PROMPT_ASSERT)


def primary_op(triple):
    ops = triple.get("tkgu_operations") or ["UNKNOWN"]
    return ops[0]


def count_per_op(jsonl_files, label):
    counts = Counter()
    n_files = 0
    n_records = 0
    for path in sorted(jsonl_files):
        n_files += 1
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_records += 1
                for t in rec.get("tkgu_triples", []):
                    if keep_triple_rule_c(t):
                        counts[primary_op(t)] += 1
    print(f"  [{label}] {n_files} files, {n_records:,} records, {sum(counts.values()):,} kept triples", flush=True)
    return counts


def discover_jsonl(root, label):
    """Walk root and return all delta_*.jsonl files. Supports both:
      - merged_v2 layout: snapshot_*-01-01/llm_assessed/delta_*.jsonl
      - eval-set layout:  snapshot_*-01-01/delta_*.jsonl    (no llm_assessed/ subdir)
    """
    nested = sorted(root.glob("snapshot_*-01-01/llm_assessed/delta_*.jsonl"))
    flat = sorted(root.glob("snapshot_*-01-01/delta_*.jsonl"))
    files = nested if nested else flat
    if not files:
        sys.exit(f"ERROR: no delta_*.jsonl files found under {root} (tried both layouts)")
    print(f"  [{label}] found {len(files)} files via "
          f"{'nested (llm_assessed/)' if nested else 'flat'} layout under {root}",
          flush=True)
    return files


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--merged-root", type=Path, required=True,
                    help="Path to merged_v2 tree (complete dataset).")
    ap.add_argument("--eval-set-root", type=Path, required=True,
                    help="Path to evaluation_set tree (3,500 test instances).")
    ap.add_argument("--output-pdf", type=Path, required=True)
    ap.add_argument("--output-png", type=Path, required=True)
    args = ap.parse_args()

    if not args.merged_root.is_dir():
        sys.exit(f"ERROR: --merged-root not found: {args.merged_root}")
    if not args.eval_set_root.is_dir():
        sys.exit(f"ERROR: --eval-set-root not found: {args.eval_set_root}")

    print(f"[fig_tkgu] scanning complete dataset: {args.merged_root}", flush=True)
    complete_files = discover_jsonl(args.merged_root, "complete")
    complete_counts = count_per_op(complete_files, "complete")

    print(f"[fig_tkgu] scanning subsampled test set: {args.eval_set_root}", flush=True)
    sub_files = discover_jsonl(args.eval_set_root, "subsampled")
    sub_counts = count_per_op(sub_files, "subsampled")

    # Render
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    colors_pie = sns.color_palette("colorblind", 5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    pies = [
        ("complete_dataset", "Complete dataset", complete_counts, axes[0]),
        ("subsampled_dataset", "Subsampled test set", sub_counts, axes[1]),
    ]

    for key, title, counts, ax in pies:
        values = [counts.get(op, 0) for op in TKGU_ORDER]
        labels = [f"{LABEL_MAP[op]}\n{v:,}" for op, v in zip(TKGU_ORDER, values)]
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors_pie,
            labeldistance=1.0,
            pctdistance=0.8,
        )
        if len(texts) > 4:
            x, y = texts[4].get_position()
            texts[4].set_position((x - 0.2, y))
        for t in texts:
            t.set_multialignment("center")
        for at in autotexts:
            at.set_color("black")
            at.set_fontsize(9)
        ax.set_title(title)

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output_pdf), bbox_inches="tight")
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output_png), bbox_inches="tight", dpi=300)
    print(f"\n[fig_tkgu] complete:   {dict(sorted(complete_counts.items()))}")
    print(f"[fig_tkgu] subsampled: {dict(sorted(sub_counts.items()))}")
    print(f"[fig_tkgu] PDF saved:  {args.output_pdf}")
    print(f"[fig_tkgu] PNG saved:  {args.output_png}")


if __name__ == "__main__":
    main()
