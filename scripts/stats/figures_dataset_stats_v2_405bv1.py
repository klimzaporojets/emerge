#!/usr/bin/env python3
"""Regenerate three dataset-statistics figures under the 405B-prompt-v1
(rule-C) filter, replacing the legacy 8B-filter versions produced by:
  - src/stats/dataset/figure_tkgu_distribution.ipynb         (pie chart)
  - src/stats/dataset/appendix_figure_tkgu_distribution_deltas.ipynb  (deltas stacked bars)
  - src/stats/dataset/appendix_figure_annotation_stats.ipynb (all vs supported per op)

All three figures use the same filter:
  - non-d-triples: 405B-prompt-v1 with prompt_type=triple_assertion
  - d-triples:     405B-prompt-v1 with prompt_type=triple_deprecation

Single walk of merged_v2 + evaluation_set; renders 3 PDFs (+ PNGs).

Usage (typical: against the public HF release, after `./scripts/download_data.sh --corpus`):

    python scripts/stats/figures_dataset_stats_v2_405bv1.py \\
        --merged-root data/corpus \\
        --eval-set-root data/evaluation_set \\
        --output-dir output/figures

Both --merged-root and --eval-set-root accept either the flat HF layout
(snapshot_*/delta_*.jsonl, auto-detected) or an internal nested
merged-tree layout (snapshot_*/llm_assessed/delta_*.jsonl).
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
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
SHORT_LABEL = {
    "x-triples": "E",
    "d-triples": "D",
    "e-triples": "A",
    "ee-triples": "M+A",
    "ee-kg-triples": "I",
}
SHORT_ORDER = ["E", "I", "A", "M+A", "D"]

# delta filename → delta_weeks: Jan 8 = 1 wk after Jan 1, Jan 15 = 2 wk, etc.
DELTA_WEEKS_FROM_FILENAME = {
    "01-08": 1, "01-15": 2, "01-22": 3, "01-29": 4, "02-05": 5,
}
DELTA_RE = re.compile(r"delta_\d{4}-(\d{2}-\d{2})\.jsonl")


def primary_op(triple):
    ops = triple.get("tkgu_operations") or ["UNKNOWN"]
    return ops[0]


def expected_prompt_type_for_op(op):
    return PROMPT_DEP if op == "d-triples" else PROMPT_ASSERT


def find_v1_entry(triple, prompt_type):
    """Find the 405B-v1 llm_assessment entry matching this prompt_type.
    Returns the entry dict or None.
    """
    for a in triple.get("llm_assessment", []):
        if a.get("llm_name") == LLM_405B_V1 and a.get("llm_prompt_type") == prompt_type:
            return a
    return None


def get_delta_weeks(path):
    m = DELTA_RE.search(path.name)
    if not m:
        return None
    return DELTA_WEEKS_FROM_FILENAME.get(m.group(1))


def discover_jsonl(root):
    """Walk root, return all delta_*.jsonl files. Supports nested
    (snapshot_*/llm_assessed/delta_*.jsonl, merged_v2 layout) and flat
    (snapshot_*/delta_*.jsonl, evaluation_set layout)."""
    nested = sorted(root.glob("snapshot_*-01-01/llm_assessed/delta_*.jsonl"))
    flat = sorted(root.glob("snapshot_*-01-01/delta_*.jsonl"))
    files = nested if nested else flat
    if not files:
        sys.exit(f"ERROR: no delta_*.jsonl files found under {root}")
    return files


def aggregate(files, label):
    """For one tree (complete or subsampled), walk all delta_*.jsonl
    and per (record, triple) compute:
      - per-op all-matched count    (has_v1_verdict)
      - per-op supported count      (v1 verdict == True)
      - per (delta_weeks, op) supported count
    """
    matched_per_op = Counter()
    supported_per_op = Counter()
    supported_per_delta_op = defaultdict(int)   # {(delta_weeks, op): n}
    n_records = 0
    for path in files:
        delta_weeks = get_delta_weeks(path)
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_records += 1
                for t in rec.get("tkgu_triples", []):
                    op = primary_op(t)
                    pt = expected_prompt_type_for_op(op)
                    entry = find_v1_entry(t, pt)
                    if entry is None:
                        continue
                    val = entry.get("llm_assessment")
                    if isinstance(val, bool):
                        matched_per_op[op] += 1
                        if val is True:
                            supported_per_op[op] += 1
                            if delta_weeks is not None:
                                supported_per_delta_op[(delta_weeks, op)] += 1
    print(f"  [{label}] {len(files)} files, {n_records:,} records, "
          f"matched={sum(matched_per_op.values()):,} supported={sum(supported_per_op.values()):,}")
    return {
        "matched_per_op": dict(matched_per_op),
        "supported_per_op": dict(supported_per_op),
        "supported_per_delta_op": dict(supported_per_delta_op),
    }


# ---------------------------------------------------------------------
# Figure renderers
# ---------------------------------------------------------------------

def render_pie(stats_complete, stats_sub, output_pdf, output_png):
    """1x2 pies of supported-per-op."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    colors_pie = sns.color_palette("colorblind", 5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    pies = [
        ("Complete dataset", stats_complete["supported_per_op"], axes[0]),
        ("Subsampled test set", stats_sub["supported_per_op"], axes[1]),
    ]
    for title, sup_per_op, ax in pies:
        values = [sup_per_op.get(op, 0) for op in TKGU_ORDER]
        labels = [f"{LABEL_MAP[op]}\n{v:,}" for op, v in zip(TKGU_ORDER, values)]
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct="%1.1f%%",
            startangle=90, colors=colors_pie, labeldistance=1.0, pctdistance=0.8,
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

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_pdf), bbox_inches="tight")
    fig.savefig(str(output_png), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  pie  PDF: {output_pdf}")
    print(f"  pie  PNG: {output_png}")


def render_deltas(stats_complete, stats_sub, output_pdf, output_png):
    """1x2 stacked bars of supported-per-(delta_weeks, op)."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    runs = [
        ("Complete dataset", stats_complete["supported_per_delta_op"], 0),
        ("Subsampled test set", stats_sub["supported_per_delta_op"], 1),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    colors_bar = plt.get_cmap("Paired").colors[: len(TKGU_ORDER)]
    delta_weeks_range = [1, 2, 3, 4, 5]

    for title, sup_per_dw_op, idx in runs:
        ax = axes[idx]
        # Build stacked counts: rows = delta_weeks, cols = op (in TKGU_ORDER)
        counts = {dw: {op: 0 for op in TKGU_ORDER} for dw in delta_weeks_range}
        for (dw, op), n in sup_per_dw_op.items():
            if dw in counts and op in counts[dw]:
                counts[dw][op] = n

        x = np.arange(len(delta_weeks_range))
        bottom = np.zeros(len(delta_weeks_range))
        for i, op in enumerate(TKGU_ORDER):
            values = np.array([counts[dw][op] for dw in delta_weeks_range], dtype=float)
            label = LABEL_MAP[op]
            ax.bar(x, values, bottom=bottom, color=colors_bar[i],
                   edgecolor="black", linewidth=0.6, width=0.4, label=label)
            for j, (val, btm) in enumerate(zip(values, bottom)):
                if j == 0 and idx == 0:
                    continue
                if val > 0:
                    total = sum(counts[delta_weeks_range[j]][o] for o in TKGU_ORDER)
                    pct = (val / total * 100) if total else 0.0
                    ax.text(j, btm + val / 2, f"{pct:.1f}%", ha="center",
                            va="center", fontsize=9, color="black")
            bottom += values

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(delta_weeks_range, fontsize=10)
        ax.set_xlabel("Δ Weeks", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Number of TKGU operations", fontsize=10)
            handles, labels_leg = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels_leg[::-1], title="TKGU operation type",
                      fontsize=10, title_fontsize=9, frameon=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.tick_params(width=0.8, labelsize=10)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{int(x/1000)}K" if x >= 1000 else f"{int(x)}")
        )

    plt.subplots_adjust(wspace=0.2)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_pdf), bbox_inches="tight")
    fig.savefig(str(output_png), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  delta PDF: {output_pdf}")
    print(f"  delta PNG: {output_png}")


def render_annotation_stats(stats_complete, stats_sub, output_pdf, output_png):
    """1x2 grouped bars: 'All' (matched) vs 'Supported' (True) per op label.
    Mirrors the canonical layout from
    wikidata-temp/wikipedia-temp/src/stats/s14_stats_dataset_llm_supported_vs_all.ipynb.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    colors = ["#1b9e77", "#d95f02"]  # All, Supported

    runs = [
        ("Complete dataset", stats_complete, 0),
        ("Subsampled test set", stats_sub, 1),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=False)

    op_for_label = {SHORT_LABEL[op]: op for op in TKGU_ORDER}
    ops_in_order = [op_for_label[lbl] for lbl in SHORT_ORDER]

    for title, stats, idx in runs:
        ax = axes[idx]
        matched = stats["matched_per_op"]
        supported = stats["supported_per_op"]

        m_vals = np.array([matched.get(op, 0) for op in ops_in_order])
        s_vals = np.array([supported.get(op, 0) for op in ops_in_order])

        x = np.arange(len(SHORT_ORDER))
        w = 0.35
        bars_all = ax.bar(x - w/2, m_vals, w, label="All", color=colors[0])
        bars_supported = ax.bar(x + w/2, s_vals, w, label="Supported", color=colors[1])

        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_ORDER, rotation=0)

        # Y-axis: "K" formatter + dashed gray grid + 6 ticks max
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y/1000)}K"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

        # Fraction label above supported bars: supported / matched
        for i, bar in enumerate(bars_supported):
            m_i = int(m_vals[i])
            s_i = int(s_vals[i])
            frac = (s_i / m_i) if m_i else 0.0
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{frac:.2f}", ha="center", va="bottom", fontsize=10)

        ax.legend()

    axes[0].set_ylabel("Number of TKGU operations", fontsize=11)
    fig.text(0.5, 0.00, "TKGU operation type (prefix)", ha="center", va="center", fontsize=12)
    plt.tight_layout()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_pdf), bbox_inches="tight")
    fig.savefig(str(output_png), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  annot PDF: {output_pdf}")
    print(f"  annot PNG: {output_png}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--merged-root", type=Path, required=True,
                    help="Path to merged_v2 tree (complete dataset).")
    ap.add_argument("--eval-set-root", type=Path, required=True,
                    help="Path to evaluation_set tree (3,500 test instances).")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Output directory for PDFs and PNGs.")
    ap.add_argument("--output-suffix", type=str, default="",
                    help="Optional suffix appended to each output basename "
                         "BEFORE the extension. Use to avoid overwriting "
                         "earlier renders. Example: '_v2' produces "
                         "pie_chart_nr_triples_both_405bv1_v2.pdf etc. "
                         "Default empty.")
    args = ap.parse_args()

    if not args.merged_root.is_dir():
        sys.exit(f"ERROR: --merged-root not found: {args.merged_root}")
    if not args.eval_set_root.is_dir():
        sys.exit(f"ERROR: --eval-set-root not found: {args.eval_set_root}")

    print(f"[fig_v2_405bv1] aggregating complete: {args.merged_root}")
    complete_files = discover_jsonl(args.merged_root)
    stats_complete = aggregate(complete_files, "complete")

    print(f"[fig_v2_405bv1] aggregating subsampled: {args.eval_set_root}")
    eval_files = discover_jsonl(args.eval_set_root)
    stats_sub = aggregate(eval_files, "subsampled")

    sx = args.output_suffix
    print()
    print("[fig_v2_405bv1] rendering pie chart ...")
    render_pie(stats_complete, stats_sub,
               args.output_dir / f"pie_chart_nr_triples_both_405bv1{sx}.pdf",
               args.output_dir / f"pie_chart_nr_triples_both_405bv1{sx}.png")
    print()
    print("[fig_v2_405bv1] rendering deltas figure ...")
    render_deltas(stats_complete, stats_sub,
                  args.output_dir / f"tkgu_distribution_deltas_405bv1{sx}.pdf",
                  args.output_dir / f"tkgu_distribution_deltas_405bv1{sx}.png")
    print()
    print("[fig_v2_405bv1] rendering annotation_stats figure ...")
    render_annotation_stats(stats_complete, stats_sub,
                            args.output_dir / f"annotation_stats_405bv1{sx}.pdf",
                            args.output_dir / f"annotation_stats_405bv1{sx}.png")

    print()
    print("[fig_v2_405bv1] === SUMMARY ===")
    print(f"  complete supported per op:   {stats_complete['supported_per_op']}")
    print(f"  subsampled supported per op: {stats_sub['supported_per_op']}")


if __name__ == "__main__":
    main()
