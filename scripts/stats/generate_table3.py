#!/usr/bin/env python3
"""
Generate the main results table (paper Table 3: Completeness + G-BERTScore-R per
TKGU operation) from a `wiki_eval_result.pkl`, and optionally verify it against
the published paper numbers.

Reviewer-facing, reproducible CLI — no notebook needed. It applies the DOCUMENTED
aggregation used for the paper table, via `src/stats/evaluation/load_results.py`:

    1. keep gold triples flagged text-supported by the 405B assessor
       (triple_assertion / triple_deprecation -> Meta-Llama-3.1-405B_prompt_v1);
    2. threshold Completeness at a strict > 0.90 (fraction of gold above threshold);
    3. per (TKGU operation, model) mean, x100.

Usage (run from the emerge repo root, after ./scripts/run/evaluate.sh has produced the pkl):

    PYTHONPATH="$PWD/src" python scripts/stats/generate_table3.py \
        --pkl output/s0x_evaluate_predictions/20260324_all_models_with_zs_fixed_with_kg/wiki_eval_result.pkl \
        --compare

`--compare` prints PASS/FAIL vs the published Table 3 (tolerance 0.05 for rounding).
Without `--compare` it just prints the readable table + a LaTeX body.

Note: this script covers Completeness (C) and G-BERTScore-R (G-R). Executable-R
(E-R; ReLiK cIE only, Exists/Add) comes from the cie_exact_match metric — see
`src/stats/evaluation/table_cie_exact_match.ipynb`.
"""
import argparse

import pandas as pd

from stats.evaluation.load_results import (
    load_results,
    models_to_load,
    model_name_to_latex,
)

ASSESSORS = {
    'triple_assertion': 'Meta-Llama-3.1-405B_prompt_v1',
    'triple_deprecation': 'Meta-Llama-3.1-405B_prompt_v1',
}

TKGU_ORDER = ['x-triples', 'e-triples', 'ee-triples', 'ee-kg-triples', 'd-triples']
TKGU_NAMES = {
    'x-triples': 'Exists', 'e-triples': 'Add', 'ee-triples': 'Mint+Add',
    'ee-kg-triples': 'Infer', 'd-triples': 'Deprecate',
}

# Fixed row order (matches the paper table).
MODEL_ORDER = [
    'edc-plus-open-ai/gpt-5.1/non-canonicalized',
    'edc-plus-azure_ai/Mistral-Large-2411',
    'edc-plus-azure_ai/Mistral-small',
    'kg-gen/azure/gpt5.1',
    'kg-gen/azure_ai/Mistral-Large-2411',
    'kg-gen/azure_ai/Mistral-small',
    'rakg/azure_ai/Mistral-Large-2411',
    'rakg/azure_ai/Mistral-small',
    'relik-oie',
    'relik-cie',
    'rebel',
]

# metric name in pkl, evaluator model, short label
METRICS = [
    ('completeness', 'all-mpnet-base-v2', 'C'),
    ('gj-recall', 'bert-base', 'G-R'),
]

# Published Table 3 targets: (model, tkgu) -> (C, G-R). None -> "--" (op not supported).
TARGETS = {
    'edc-plus-open-ai/gpt-5.1/non-canonicalized': {
        'x-triples': (25.7, 66.3), 'e-triples': (40.2, 77.5), 'ee-triples': (35.6, 77.3),
        'ee-kg-triples': (21.9, 72.4), 'd-triples': (34.8, 53.2)},
    'edc-plus-azure_ai/Mistral-Large-2411': {
        'x-triples': (16.1, 53.3), 'e-triples': (30.3, 70.7), 'ee-triples': (27.9, 71.5),
        'ee-kg-triples': (19.9, 68.8), 'd-triples': (31.7, 53.1)},
    'edc-plus-azure_ai/Mistral-small': {
        'x-triples': (8.8, 45.4), 'e-triples': (21.1, 63.5), 'ee-triples': (16.3, 62.6),
        'ee-kg-triples': (21.4, 70.8), 'd-triples': (14.5, 28.6)},
    'kg-gen/azure/gpt5.1': {
        'x-triples': (12.8, 62.7), 'e-triples': (15.2, 68.1), 'ee-triples': (13.3, 67.0),
        'ee-kg-triples': (3.3, 59.1), 'd-triples': None},
    'kg-gen/azure_ai/Mistral-Large-2411': {
        'x-triples': (11.8, 60.4), 'e-triples': (16.0, 67.7), 'ee-triples': (13.7, 65.9),
        'ee-kg-triples': (2.1, 56.9), 'd-triples': None},
    'kg-gen/azure_ai/Mistral-small': {
        'x-triples': (8.1, 57.0), 'e-triples': (13.0, 65.8), 'ee-triples': (12.6, 65.3),
        'ee-kg-triples': (2.1, 56.1), 'd-triples': None},
    'rakg/azure_ai/Mistral-Large-2411': {
        'x-triples': (11.7, 64.1), 'e-triples': (19.0, 70.9), 'ee-triples': (16.5, 70.7),
        'ee-kg-triples': (2.5, 62.6), 'd-triples': None},
    'rakg/azure_ai/Mistral-small': {
        'x-triples': (11.7, 64.1), 'e-triples': (19.2, 71.1), 'ee-triples': (16.0, 70.7),
        'ee-kg-triples': (2.5, 62.6), 'd-triples': None},
    'relik-oie': {
        'x-triples': (21.9, 59.4), 'e-triples': (20.6, 61.8), 'ee-triples': (17.1, 60.9),
        'ee-kg-triples': (3.8, 48.8), 'd-triples': None},
    'relik-cie': {
        'x-triples': (15.2, 30.7), 'e-triples': (18.1, 57.5), 'ee-triples': None,
        'ee-kg-triples': None, 'd-triples': None},
    'rebel': {
        'x-triples': (12.3, 43.9), 'e-triples': (13.6, 56.3), 'ee-triples': (11.6, 55.1),
        'ee-kg-triples': (1.5, 37.6), 'd-triples': None},
}

TOL = 0.05  # rounding tolerance for a cell to count as a match


def compute(pkl_path):
    """Return {(model, tkgu, short): value} using the documented aggregation."""
    (df_cie, _open, _gt, _p, _inst, _add) = load_results(
        pkl_path,
        assessor_by_prompt_type=ASSESSORS,
        filter_models=set(models_to_load),
    )
    df = df_cie.copy()
    if 'triple_type' in df.columns:
        df = df[df['triple_type'] == 'in-dataset']
    df['score'] = df['score'].astype(float)

    results = {}
    for metric, eval_model, short in METRICS:
        mask = (df['metric'] == metric) & (df['evaluator_model'] == eval_model)
        agg = df[mask].groupby(['tkgu_type', 'model'])['score'].mean() * 100.0
        for (tkgu, model), score in agg.items():
            results[(model, tkgu, short)] = float(score)
    return results


def _cell(results, model, tkgu, short):
    v = results.get((model, tkgu, short), 0.0)
    # A model that does not support an op produces no predictions -> all scores 0.
    return v if v > 0 else None


def print_table(results):
    print("\n=== Table 3 (Completeness C, G-BERTScore-R G-R) — produced from pkl ===\n")
    hdr = f"{'Model':<24}"
    for t in TKGU_ORDER:
        hdr += f" | {TKGU_NAMES[t]:>9} C   G-R"
    print(hdr)
    print("-" * len(hdr))
    for m in MODEL_ORDER:
        row = f"{model_name_to_latex.get(m, m):<24}"
        for t in TKGU_ORDER:
            c = _cell(results, m, t, 'C')
            gr = _cell(results, m, t, 'G-R')
            row += (f" | {c:>11.1f} {gr:>5.1f}" if c is not None
                    else f" | {'--':>11} {'--':>5}")
        print(row)


def compare(results):
    print("\n=== Compare vs published Table 3 (tol %.2f) ===\n" % TOL)
    n_ok = n_bad = 0
    fails = []
    for m in MODEL_ORDER:
        for t in TKGU_ORDER:
            tgt = TARGETS[m][t]
            got_c = _cell(results, m, t, 'C')
            got_gr = _cell(results, m, t, 'G-R')
            if tgt is None:
                # op unsupported -> expect no value
                ok = got_c is None
                if ok:
                    n_ok += 1
                else:
                    n_bad += 1
                    fails.append(f"{model_name_to_latex.get(m,m)} / {TKGU_NAMES[t]}: "
                                 f"expected '--' but produced C={got_c:.1f}")
                continue
            tc, tgr = tgt
            for label, got, exp in (('C', got_c, tc), ('G-R', got_gr, tgr)):
                if got is not None and abs(got - exp) <= TOL:
                    n_ok += 1
                else:
                    n_bad += 1
                    g = 'None' if got is None else f'{got:.1f}'
                    fails.append(f"{model_name_to_latex.get(m,m)} / {TKGU_NAMES[t]} {label}: "
                                 f"expected {exp:.1f}, produced {g}")
    for f in fails:
        print("  FAIL:", f)
    print(f"\n  {n_ok} cells OK, {n_bad} cells FAIL")
    print("  ==> " + ("✅ 100% MATCH — reproduction verified"
                       if n_bad == 0 else "❌ MISMATCH — do NOT proceed to building; debug first"))
    return n_bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True, help='path to wiki_eval_result.pkl')
    ap.add_argument('--compare', action='store_true',
                    help='diff each cell vs the published Table 3 and print PASS/FAIL')
    args = ap.parse_args()

    results = compute(args.pkl)
    print_table(results)
    if args.compare:
        ok = compare(results)
        raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
