#!/usr/bin/env python3
"""SCORE the KG-conditioned baseline (kg_prompt) ladder with the paper's Table-3 metrics.

Three phases, all driven by one config (config/kg_context/20260725_kg_prompt_ladder_score/config.json):

  merge      For each (op, condition) point, inject that point's kg_prompt predictions
             (keyed by hash_id, KGGen OIE schema: predicted_triples=[{extracted_relation,
             triple_qids, triple_labels}]) into that point's COMPOSED eval-set (which already
             carries the gold `tkgu_triples` + 405B `llm_assessment`) under alias `kg-prompt`
             / model_type `kg-gen`. Writes a scoreable per-point tree. No GPU / no LLM.

  score      For each point, run `python -m evaluation.s0x_evaluate_predictions` UNCHANGED,
             overriding input_dataset_path / output_path / cache_path per point on the CLI.
             GPU + KG snapshots — run on a Snellius compute node, NOT the login node. One run
             per point is REQUIRED: the preloader asserts each hash_id appears once, and the
             same passage recurs across conditions, so points cannot share a single tree.

  aggregate  Read every per-point pkl and print/write the ladder table: rows = (point, TKGU op),
             columns = C, G-BERTScore-R (G-R), Executable-R (E-R). Metrics are UNCHANGED from
             Table 3:
               C   = metric 'completeness'   / evaluator 'all-mpnet-base-v2' (thresholded >0.9)
               G-R = metric 'gj-recall'       / evaluator 'bert-base'   (graph_scorers, BERTScore)
               E-R = metric 'cie-recall'      / evaluator 'exact_match' (cie_exact_match_scorer)
             Aggregation (405B-assessor GT filter, >0.9 completeness threshold, per-(op) mean x100)
             is REUSED from stats.evaluation.load_results — identical to scripts/stats/generate_table3.py.

  all        merge -> score -> aggregate.

Usage (from emerge repo root, PYTHONPATH=src):
    PYTHONPATH=src python scripts/kg_context/score_kg_prompt.py --config <cfg> --phase merge
    PYTHONPATH=src python scripts/kg_context/score_kg_prompt.py --config <cfg> --phase score
    PYTHONPATH=src python scripts/kg_context/score_kg_prompt.py --config <cfg> --phase aggregate
    PYTHONPATH=src python scripts/kg_context/score_kg_prompt.py --config <cfg> --phase all
Add --dry-run to preview merge/score actions without writing/executing.

NOTE on E-R: `kg-prompt` is merged as model_type `kg-gen` (an OIE model). The preloader's OIE
path scores C and G-R from predicted surface triples; it does NOT canonicalise QIDs, so E-R is
0 / "--" for it — exactly the treatment every OIE baseline (KGGen/RAKG/REBEL/ReLiK-RE) gets in
Table 3, where Executable-R is meaningful only for the fully-canonicalised ReLiK cIE. E-R is
still computed & reported for parity. To make E-R meaningful for a QID-carrying variant you
would merge under a model_type whose preloader branch populates predicted_triples_cie (relik /
llm-tool) — that is a deliberate, separate choice, not a silent default here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

TKGU_ORDER = ['x-triples', 'e-triples', 'ee-triples', 'ee-kg-triples', 'd-triples']
TKGU_NAMES = {
    'x-triples': 'Exists', 'e-triples': 'Add', 'ee-triples': 'Mint+Add',
    'ee-kg-triples': 'Infer', 'd-triples': 'Deprecate',
}

# (metric name in pkl, evaluator model, short label) — SAME keys as Table 3.
METRICS = [
    ('completeness', 'all-mpnet-base-v2', 'C'),
    ('gj-recall', 'bert-base', 'G-R'),
    ('cie-recall', 'exact_match', 'E-R'),
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def _resolve_points(config: dict) -> list:
    """The condition-cells to score (one aggregated C/G-R/E-R row each; every cell holds all N instances).
    SOURCE OF TRUTH = the COMPOSE manifest (`<gold_base_dir>/manifest.json`), so the cell list can never drift
    from what was actually composed. `config['points']` may still pin an EXPLICIT subset (e.g. the smoke cells)
    -- a non-empty list wins; null / [] / absent -> derive ALL cells from the manifest."""
    override = config.get('points')
    if isinstance(override, list) and override:
        print(f'[points] using {len(override)} explicit cell(s) from config (override)')
        return override
    manifest_path = os.path.join(config['gold_base_dir'], 'manifest.json')
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f'cannot derive condition-cells: no compose manifest at {manifest_path} '
                                f'(run COMPOSE first, or pin config["points"] explicitly)')
    with open(manifest_path, 'rt', encoding='utf-8') as f:
        manifest = json.load(f)
    tags = [p['tag'] for p in manifest['points']]
    print(f'[points] derived {len(tags)} cell(s) from compose manifest {manifest_path}')
    return tags


def _iter_jsonl_files(root: str):
    """Yield (rel_path, abs_path) for every .jsonl file under root (sorted, stable)."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in sorted(filenames):
            if fname.endswith('.jsonl'):
                abs_path = os.path.join(dirpath, fname)
                yield os.path.relpath(abs_path, root), abs_path


def _read_point_predictions(pred_point_dir: str, source_pred_alias: str) -> dict:
    """Build {hash_id: prediction_entry} from a point's kg_prompt prediction tree.

    Each prediction file record is the full instance written by save_batch_predictions:
    `predictions` is a dict keyed by the wrapper model_name (`source_pred_alias`), whose value
    is {'predicted_triples': [...], 'model': ..., 'model_config_name': ...}. Legacy list form
    (exactly one entry) is also accepted.
    """
    hash_to_pred = {}
    for _rel, abs_path in _iter_jsonl_files(pred_point_dir):
        with open(abs_path, 'rt', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                hash_id = rec['hash_id']
                preds = rec['predictions']
                if isinstance(preds, dict):
                    if source_pred_alias not in preds:
                        raise KeyError(
                            f'source_pred_alias={source_pred_alias!r} not in predictions dict '
                            f'keys={list(preds.keys())} for hash_id={hash_id} in {abs_path}'
                        )
                    entry = preds[source_pred_alias]
                elif isinstance(preds, list):
                    if len(preds) != 1:
                        raise RuntimeError(
                            f'expected exactly 1 prediction in list, got {len(preds)} for '
                            f'hash_id={hash_id} in {abs_path}'
                        )
                    entry = preds[0]
                else:
                    raise RuntimeError(
                        f'unexpected predictions type {type(preds)} for hash_id={hash_id} in {abs_path}'
                    )
                if hash_id in hash_to_pred:
                    raise RuntimeError(f'duplicate hash_id in predictions: {hash_id} ({abs_path})')
                hash_to_pred[hash_id] = entry
    return hash_to_pred


# ---------------------------------------------------------------------------
# Phase 1: merge
# ---------------------------------------------------------------------------

def phase_merge(config: dict, dry_run: bool) -> None:
    gold_base = config['gold_base_dir']
    pred_base = config['pred_base_dir']
    merged_base = config['merged_base_dir']
    pred_model_subdir = config['pred_model_subdir']
    source_pred_alias = config['source_pred_alias']
    target_alias = config['target_alias']
    model_type = config['predictions_model_type']
    model_config_name = config['predictions_model_config_name']

    # OPTIONAL instance scope: when `hash_ids_file` is set, score ONLY those hash_ids (e.g. the 350-instance
    # subset) — gold records outside the set are SKIPPED (not required to have a prediction). Mirrors
    # predict_triples.py --hash-ids so gold (composed over all 3,500) and preds (the 350) line up. Explicit
    # membership check (no .get defaults).
    allow = None
    if 'hash_ids_file' in config:
        with open(config['hash_ids_file']) as f:
            allow = {ln.strip() for ln in f if ln.strip()}
        print(f'[merge] SCOPING to {len(allow)} hash_ids from {config["hash_ids_file"]}')

    # PER-OP ROUTING: in per_op mode the OPERATION is FIXED by the cell (the prompt told the model which op
    # to do), so route by the CELL's op — NOT the model's self-declared `operation` tag. The model emits that
    # tag inconsistently (100% for exists/deprecate/mint_add/infer, but ~60% MISSING for add), and
    # process_kg_prompt_predictions DROPS any triple whose tag is missing -> Add was silently deflated
    # (Add/kg_rag -> 0). Stamping the cell op makes routing robust AND matches how the paper's OIE baselines
    # are scored (their extractions reach the op bucket regardless of any self-tag). Unified points (op not
    # derivable) keep the model's tag.
    _OP_LABEL_TO_OPERATION = {'exists': 'EXISTS', 'add': 'ADD', 'mint_add': 'MINT_ADD',
                              'infer': 'INFER', 'deprecate': 'DEPRECATE'}

    for point in config['points']:
        gold_point_dir = os.path.join(gold_base, point)
        pred_point_dir = os.path.join(pred_base, point, pred_model_subdir)
        merged_point_dir = os.path.join(merged_base, point)
        _parts = point.split('/')
        cell_op = (_OP_LABEL_TO_OPERATION[_parts[1]]
                   if _parts[0] == 'per_op' and len(_parts) > 1 and _parts[1] in _OP_LABEL_TO_OPERATION
                   else None)

        if not os.path.isdir(gold_point_dir):
            raise FileNotFoundError(f'gold point dir does not exist: {gold_point_dir}')
        if not os.path.isdir(pred_point_dir):
            raise FileNotFoundError(f'pred point dir does not exist: {pred_point_dir}')

        hash_to_pred = _read_point_predictions(pred_point_dir, source_pred_alias)
        print(f'[merge] {point}: {len(hash_to_pred)} predictions loaded from {pred_point_dir}')

        n_records = 0
        n_matched = 0
        for rel, abs_gold in _iter_jsonl_files(gold_point_dir):
            out_path = os.path.join(merged_point_dir, rel)
            lines_out = []
            with open(abs_gold, 'rt', encoding='utf-8') as f:
                for line in f:
                    rec = json.loads(line)
                    hash_id = rec['hash_id']
                    if allow is not None and hash_id not in allow:
                        continue                      # out of scope (not in the 350) -> skip, no pred required
                    n_records += 1
                    if 'predictions' not in rec or not isinstance(rec['predictions'], dict):
                        rec['predictions'] = {}
                    if hash_id not in hash_to_pred:
                        raise KeyError(
                            f'gold hash_id={hash_id} has no kg_prompt prediction '
                            f'(point={point}, file={rel})'
                        )
                    src_entry = hash_to_pred[hash_id]
                    pred_triples = src_entry['predicted_triples']
                    if cell_op is not None:   # per-op: route by the cell's KNOWN op, not the model's self-tag
                        pred_triples = [{**t, 'operation': cell_op} for t in pred_triples]
                    rec['predictions'][target_alias] = {
                        'predicted_triples': pred_triples,
                        'model': src_entry['model'],
                        'model_config_name': model_config_name,
                        'model_type': model_type,
                    }
                    n_matched += 1
                    lines_out.append(json.dumps(rec, ensure_ascii=False))
            if dry_run:
                print(f'[merge][dry-run] would write {len(lines_out)} records -> {out_path}')
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'wt', encoding='utf-8') as outf:
                outf.write('\n'.join(lines_out) + ('\n' if lines_out else ''))
        print(f'[merge] {point}: {n_matched}/{n_records} gold records got a `{target_alias}` '
              f'prediction -> {merged_point_dir}')


# ---------------------------------------------------------------------------
# Phase 2: score (drives s0x_evaluate_predictions UNCHANGED, one run per point)
# ---------------------------------------------------------------------------

def _point_pkl_path(config: dict, point: str) -> str:
    return os.path.join(config['pkl_base_dir'], point, 'wiki_eval_result.pkl')


def phase_score(config: dict, dry_run: bool) -> None:
    merged_base = config['merged_base_dir']
    eval_config_file = config['eval_config_file']
    module = config['python_module_score']

    points = config['points']
    n_points = len(points)
    for i, point in enumerate(points, 1):
        merged_point_dir = os.path.join(merged_base, point)
        pkl_path = _point_pkl_path(config, point)
        if not dry_run and not os.path.isdir(merged_point_dir):
            raise FileNotFoundError(
                f'merged point dir does not exist (run phase=merge first): {merged_point_dir}'
            )
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        cmd = [
            sys.executable, '-m', module,
            '--config_file', eval_config_file,
            '--input_dataset_path', merged_point_dir + os.sep,
            '--output_path', pkl_path,
            '--cache_path', pkl_path,
        ]
        print(f'[score] point {i}/{n_points}: {point}\n  ' + ' '.join(cmd))
        if dry_run:
            continue
        env = dict(os.environ)
        src_dir = os.path.abspath('src')
        env['PYTHONPATH'] = src_dir + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        subprocess.run(cmd, check=True, env=env)
        print(f'[score] point {i}/{n_points} DONE -> {pkl_path}')
    print(f'[score] === all {n_points} point(s) scored. NOTE: SCORE is not finished until the '
          f'aggregate phase writes the ladder table (run --phase aggregate, or --phase all). ===')


# ---------------------------------------------------------------------------
# Phase 3: aggregate (REUSES stats.evaluation.load_results — same as generate_table3.py)
# ---------------------------------------------------------------------------

def _compute_point(pkl_path: str, target_alias: str, assessors: dict) -> dict:
    """Return {(tkgu_op, short): value} for one point's pkl, using the documented aggregation."""
    from stats.evaluation.load_results import load_results

    (df_cie, _open, _gt, _p, _inst, _add) = load_results(
        pkl_path,
        assessor_by_prompt_type=assessors,
        filter_models={target_alias},
    )
    df = df_cie.copy()
    if 'triple_type' in df.columns:
        df = df[df['triple_type'] == 'in-dataset']
    df['score'] = df['score'].astype(float)

    out = {}
    for metric, eval_model, short in METRICS:
        mask = (df['metric'] == metric) & (df['evaluator_model'] == eval_model) & (df['model'] == target_alias)
        agg = df[mask].groupby('tkgu_type')['score'].mean() * 100.0
        for tkgu, score in agg.items():
            out[(tkgu, short)] = float(score)
    return out


def _rows_for_model(pkl_path: str, alias: str, point_label: str, assessors: dict) -> list:
    """Per-op C/G-R/E-R rows for one model from one pkl."""
    res = _compute_point(pkl_path, alias, assessors)
    rows = []
    for tkgu in TKGU_ORDER:
        c, gr, er = res.get((tkgu, 'C')), res.get((tkgu, 'G-R')), res.get((tkgu, 'E-R'))
        if c is None and gr is None and er is None:
            continue
        rows.append({'model': alias, 'point': point_label, 'tkgu_op': tkgu,
                     'tkgu_name': TKGU_NAMES[tkgu], 'C': c, 'G-R': gr, 'E-R': er})
    return rows


def phase_aggregate(config: dict, dry_run: bool) -> None:
    import csv

    target_alias = config['target_alias']
    assessors = config['assessors']
    out_csv = config['aggregate_output_csv']
    # published baselines to report alongside kg-prompt (condition-independent: their
    # prediction is fixed, so we score them ONCE from any cell's pkl). e.g. EDC+ gpt-5.1.
    baseline_aliases = config['baseline_aliases'] if 'baseline_aliases' in config else []

    rows = []
    first_pkl = None
    for point in config['points']:
        pkl_path = _point_pkl_path(config, point)
        if not os.path.exists(pkl_path):
            print(f'[aggregate] SKIP {point}: pkl not found ({pkl_path})')
            continue
        if first_pkl is None:
            first_pkl = pkl_path
        rows += _rows_for_model(pkl_path, target_alias, point, assessors)

    # baselines — once, from the first available pkl (same 32 instances in every cell)
    base_rows = []
    for base in baseline_aliases:
        if first_pkl is None:
            break
        base_rows += _rows_for_model(first_pkl, base, '(baseline: cond-indep)', assessors)

    def _fmt(v):
        return '--' if v is None else f'{v:7.1f}'

    print('\n=== KG-conditioned baseline (kg-prompt) ladder — Table-3 metrics ===')
    print(f'{"point":<30} {"op":<10} {"C":>7} {"G-R":>7} {"E-R":>7}')
    print('-' * 66)
    for r in rows:
        print(f'{r["point"]:<30} {r["tkgu_name"]:<10} {_fmt(r["C"])} {_fmt(r["G-R"])} {_fmt(r["E-R"])}')
    if base_rows:
        print('\n--- published baseline(s), same 32 instances (condition-independent) ---')
        for r in base_rows:
            print(f'{r["model"][:30]:<30} {r["tkgu_name"]:<10} {_fmt(r["C"])} {_fmt(r["G-R"])} {_fmt(r["E-R"])}')

    all_rows = rows + base_rows
    if dry_run:
        print(f'[aggregate][dry-run] would write {len(all_rows)} rows -> {out_csv}')
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'wt', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'point', 'tkgu_op', 'tkgu_name', 'C', 'G-R', 'E-R'])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f'\n[aggregate] wrote {len(all_rows)} rows -> {out_csv}')
    print(f'=== SCORE COMPLETE — ladder table written to {out_csv} ===')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='path to the ladder-score config JSON')
    ap.add_argument('--phase', required=True, choices=['merge', 'score', 'aggregate', 'all'])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    config = _load_config(args.config)
    config['points'] = _resolve_points(config)   # manifest-derived cells (or explicit override); single source of truth

    if args.phase in ('merge', 'all'):
        phase_merge(config, args.dry_run)
    if args.phase in ('score', 'all'):
        phase_score(config, args.dry_run)
    if args.phase in ('aggregate', 'all'):
        phase_aggregate(config, args.dry_run)


if __name__ == '__main__':
    main()
