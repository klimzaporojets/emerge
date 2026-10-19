#!/usr/bin/env python3
"""Aggregate the HF-validation score pkl into a per-op ladder and (optionally) diff it
against the published KG-aware ladder.

This is the AGGREGATE step of the end-to-end validation. It reads the SINGLE pkl produced by
one `s04_evaluate_predictions` pass over the merged HF file
(output/evaluation_set_with_kg_aware/) scored with config
config/evaluation/s04_evaluate_predictions/20260803_kg_aware_hf_validation/config.json, and
for each of the 3 KG-aware model keys prints C / G-BERTScore-R / Executable-R per TKGU op —
using the SAME aggregation as scripts/kg_context/score_kg_prompt.py (stats.evaluation.load_results,
405B-assessor GT filter, >0.9 completeness threshold, per-op mean x100). If --compare-ladder is
given (the published output/kg_prompts_scored/20260727_native_3500/ladder_table.csv), it maps each
(variant, op) to that ladder's (point, op) and prints the delta — a match proves the consolidated
per-instance HF file reproduces the ladder, i.e. the format is faithful end-to-end.

Usage (PYTHONPATH=src; CPU-only, reads the pkl — no GPU needed):
    PYTHONPATH=src python scripts/kg_context/aggregate_kg_aware_hf_ladder.py \
        --pkl   output/kg_prompts_scored/20260803_hf_validation/wiki_eval_result.pkl \
        --out-csv output/kg_prompts_scored/20260803_hf_validation/hf_ladder.csv \
        [--compare-ladder output/kg_prompts_scored/20260727_native_3500/ladder_table.csv] \
        [--tol 0.15]
"""
from __future__ import annotations

import argparse
import csv
import os

TKGU_ORDER = ['x-triples', 'e-triples', 'ee-triples', 'ee-kg-triples', 'd-triples']
TKGU_NAMES = {'x-triples': 'Exists', 'e-triples': 'Add', 'ee-triples': 'Mint+Add',
              'ee-kg-triples': 'Infer', 'd-triples': 'Deprecate'}

# (metric name in pkl, evaluator model, short label) — SAME keys as Table 3 / score_kg_prompt.py.
METRICS = [('completeness', 'all-mpnet-base-v2', 'C'),
           ('gj-recall', 'bert-base', 'G-R'),
           ('cie-recall', 'exact_match', 'E-R')]

KG_AWARE_MODELS = ['kg-aware/gpt-5.1/oracle',
                   'kg-aware/gpt-5.1/kg_rag_32',
                   'kg-aware/gpt-5.1/oracle_kg_rag']
REFERENCE_MODEL = 'edc-plus-open-ai/gpt-5.1/non-canonicalized'

ASSESSORS = {'triple_assertion': 'Meta-Llama-3.1-405B_prompt_v1',
             'triple_deprecation': 'Meta-Llama-3.1-405B_prompt_v1'}

# variant -> (condition for x/d triple-ops, condition for e/ee/ee-kg piece-ops); to map to ladder points.
_VARIANT_COND = {
    'kg-aware/gpt-5.1/oracle':        ('oracle_triples',    'only_oracle'),
    'kg-aware/gpt-5.1/kg_rag_32':     ('kg_rag_triples_n32', 'kg_rag_k32'),
    'kg-aware/gpt-5.1/oracle_kg_rag': ('both_triples_n32',   'oracle_pieces_k32'),
}
_OP_DIR = {'x-triples': 'exists', 'e-triples': 'add', 'ee-triples': 'mint_add',
           'ee-kg-triples': 'infer', 'd-triples': 'deprecate'}
_TRIPLE_OPS = {'x-triples', 'd-triples'}


def _ladder_point(variant: str, op: str) -> str:
    triple_cond, piece_cond = _VARIANT_COND[variant]
    cond = triple_cond if op in _TRIPLE_OPS else piece_cond
    return f'per_op/{_OP_DIR[op]}/{cond}'


def _per_op(pkl_path: str, model: str) -> dict:
    """{(tkgu_op, short): value} for one model from the single pkl (documented aggregation)."""
    from stats.evaluation.load_results import load_results
    (df_cie, _open, _gt, _p, _inst, _add) = load_results(
        pkl_path, assessor_by_prompt_type=ASSESSORS, filter_models={model})
    df = df_cie.copy()
    if 'triple_type' in df.columns:
        df = df[df['triple_type'] == 'in-dataset']
    df['score'] = df['score'].astype(float)
    out = {}
    for metric, eval_model, short in METRICS:
        mask = (df['metric'] == metric) & (df['evaluator_model'] == eval_model) & (df['model'] == model)
        for tkgu, score in (df[mask].groupby('tkgu_type')['score'].mean() * 100.0).items():
            out[(tkgu, short)] = float(score)
    return out


def _fmt(v):
    return '  --  ' if v is None else f'{v:6.1f}'


def _load_published(path: str) -> dict:
    """{(point, tkgu_op): {'C':..,'G-R':..,'E-R':..}} from the published ladder CSV."""
    pub = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            def _f(x):
                x = (x or '').strip()
                return None if x in ('', '--', 'None') else float(x)
            pub[(row['point'], row['tkgu_op'])] = {'C': _f(row.get('C')), 'G-R': _f(row.get('G-R')), 'E-R': _f(row.get('E-R'))}
    return pub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True, help='the single wiki_eval_result.pkl from the HF-validation score')
    ap.add_argument('--out-csv', required=True, help='where to write the per-op ladder CSV')
    ap.add_argument('--compare-ladder', default=None, help='published ladder_table.csv to diff against')
    ap.add_argument('--tol', type=float, default=0.15, help='max abs diff (points) treated as a match')
    args = ap.parse_args()

    pub = _load_published(args.compare_ladder) if args.compare_ladder else None
    rows = []
    max_abs = 0.0
    n_cells = n_mismatch = 0

    print(f'\n=== KG-aware HF-file validation ladder (from {args.pkl}) ===')
    hdr = f'{"model":<32} {"op":<10} {"C":>6} {"G-R":>6} {"E-R":>6}'
    if pub is not None:
        hdr += f'   {"dC":>6} {"dG-R":>6} {"dE-R":>6}  point'
    print(hdr)
    print('-' * (len(hdr) + 4))

    for model in KG_AWARE_MODELS + [REFERENCE_MODEL]:
        res = _per_op(args.pkl, model)
        for op in TKGU_ORDER:
            c, gr, er = res.get((op, 'C')), res.get((op, 'G-R')), res.get((op, 'E-R'))
            if c is None and gr is None and er is None:
                continue
            row = {'model': model, 'tkgu_op': op, 'tkgu_name': TKGU_NAMES[op], 'C': c, 'G-R': gr, 'E-R': er}
            line = f'{model:<32} {TKGU_NAMES[op]:<10} {_fmt(c)} {_fmt(gr)} {_fmt(er)}'
            if pub is not None and model in _VARIANT_COND:
                pt = _ladder_point(model, op)
                ref = pub.get((pt, op))
                if ref is not None:
                    def _d(a, b):
                        return None if (a is None or b is None) else a - b
                    dC, dG, dE = _d(c, ref['C']), _d(gr, ref['G-R']), _d(er, ref['E-R'])
                    row.update({'pub_C': ref['C'], 'pub_G-R': ref['G-R'], 'pub_E-R': ref['E-R'],
                                'dC': dC, 'dG-R': dG, 'dE-R': dE, 'point': pt})
                    for d in (dC, dG, dE):
                        if d is not None:
                            max_abs = max(max_abs, abs(d))
                            n_cells += 1
                            if abs(d) > args.tol:
                                n_mismatch += 1
                    line += f'   {_fmt(dC)} {_fmt(dG)} {_fmt(dE)}  {pt}'
                else:
                    line += f'   {"(no published row)":>22}  {pt}'
            rows.append(row)
            print(line)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fields = ['model', 'tkgu_op', 'tkgu_name', 'C', 'G-R', 'E-R']
    if pub is not None:
        fields += ['pub_C', 'pub_G-R', 'pub_E-R', 'dC', 'dG-R', 'dE-R', 'point']
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {len(rows)} rows -> {args.out_csv}')

    if pub is not None:
        print(f'\n=== COMPARISON vs published ladder ({args.compare_ladder}) ===')
        print(f'cells compared: {n_cells} | max |delta|: {max_abs:.3f} pts | '
              f'cells over tol={args.tol}: {n_mismatch}')
        if n_mismatch == 0:
            print('RESULT: MATCH — the merged HF file reproduces the ladder. Format is faithful end-to-end. '
                  'Safe to upload.')
        else:
            print('RESULT: MISMATCH — investigate before uploading (consolidation/format bug, '
                  'or a real difference in what got scored).')


if __name__ == '__main__':
    main()
