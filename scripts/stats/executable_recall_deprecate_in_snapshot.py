#!/usr/bin/env python3
"""
Executable-R (E-R) column of the camera-ready Table 3, with Deprecate E-R computed only over the
text-supported gold Deprecate triples whose (s,p,o) is in the KG snapshot T_t of the instance's year.

Why: a Deprecate operation is labelled from the passage and the triple's lifespan, with no condition on
T_t, so some supported Deprecates are not in T_t (in the v1.0.1 test set: 141 of 1,628 -- 43 with an
entity minted after t, 98 whose statement was entered into Wikidata only after t). E-R applies the
predicted edit to G_t, and a triple absent from G_t cannot be deprecated, so E-R cannot evaluate these
by definition. They stay in the dataset and are scored by Completeness and G-BERTScore-R. Only the
KG-aware rows have a Deprecate E-R (no other method outputs canonical Deprecate triples); Exists / Add
E-R are unchanged. The dataset is not modified.

Steps (all settings in --config):
  1. test set: year of every instance; supported gold Deprecates (first assessment of `deprecate_assessor`
     with prompt triple_deprecation) and their emerging_head / emerging_tail flags;
  2. streams the 7 KG snapshots once (gzip, never decompressed; only those keys are looked up) -- before
     the results files are loaded, so the worker processes stay small;
  3. stops unless the counts equal `expected_counts`;
  4. KG-aware results file: re-scores the rows of the cie_exact_match metric with the same scorer
     (src/evaluation/scorers/cie_exact_match_scorer.py, exclude_empty_gold=True as in the s04 configs);
     stops unless it reproduces `expected_s04_er` and its Deprecate gold equals the test set's;
  5. Deprecate E-R with gold = supported AND in T_t (instances left without such gold are skipped);
  6. IE results file: reproduces ReLiK cIE Exists / Add E-R;
  7. stops unless every final cell equals `expected_table3_er` (tolerance 0.05).

Outputs (output_dir, never overwritten):
  er_column.json                       every E-R cell: all supported gold (s04) / gold in T_t, instance counts
  er_column_cells.txt                  the cells in Table 3 order, one decimal
  deprecate_gold_not_in_snapshot.tsv   supported gold Deprecates not in T_t (year, hash_id, s, p, o, reason)

Usage (from the repository root, after the two s04 runs named in the config):

    PYTHONPATH="$PWD/src" python scripts/stats/executable_recall_deprecate_in_snapshot.py \\
        --config config/evaluation/executable_recall_deprecate_in_snapshot/20261030_v1_0_1_reissue/config.json

Needs no GPU; reads the two results files (~1.1 GB together) and streams the snapshots (~3.8 GB, ~1-2 min
with 7 processes). Test: tests/test_executable_recall_deprecate_in_snapshot.py.
"""
import argparse
import gc
import glob
import json
import os
import pickle
import time
from multiprocessing import Pool

from evaluation.scorers.cie_exact_match_scorer import calculate_cie_exact_match
from kg_context.snapshot_io import open_kg_snapshot

YEARS = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']
TOLERANCE = 0.05
T0 = time.time()


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")} +{time.time() - T0:6.0f}s] {msg}', flush=True)


def mean_recall(rows, model, tkgu):
    """Mean per-instance cie-recall x100 with the s04 scorer (exclude_empty_gold=True, as in the s04 configs)."""
    scores = calculate_cie_exact_match(rows, model=model, tkgu_type=tkgu, exclude_empty_gold=True)
    rec = [r['score'] for r in scores['scores_per_instance'] if r['metric'] == 'cie-recall']
    return 100 * sum(rec) / len(rec), len(rec)


def supported(triple, assessor):
    """First assessment of `assessor` with prompt triple_deprecation; True only if True."""
    for a in triple['llm_assessment']:
        if a['llm_name'] == assessor and a['llm_prompt_type'] == 'triple_deprecation':
            return a['llm_assessment'] is True
    return False


def restrict_to_snapshot(rows, in_snapshot, year_of):
    """Rows [hash_id, gold, pred, flags] -> same rows with flags = flag AND (s,p,o) in T_t of the instance's year."""
    return [[r[0], r[1], r[2], [bool(ok) and tuple(t) in in_snapshot[year_of[r[0]]] for t, ok in zip(r[1], r[3])]]
            for r in rows]


def find_in_snapshot(job):
    """Stream one snapshot; return which of the wanted (s, p, o) keys it contains."""
    year, path, wanted = job
    found, n, last = set(), 0, time.time()
    with open_kg_snapshot(path) as f:
        for line in f:
            s, p, o, _ = line.split('\t', 3)
            if (s, p, o) in wanted:
                found.add((s, p, o))
            n += 1
            if time.time() - last > 30:
                log(f'{year}: {n:,} lines read, {len(found)}/{len(wanted)} found')
                last = time.time()
    log(f'{year}: done, {n:,} lines, {len(found)}/{len(wanted)} found')
    return year, found


def load_batch(path):
    log(f'loading {path}')
    with open(path, 'rb') as f:
        batch = pickle.load(f).batch_cie_qid_triples
    log('loaded')
    return batch


def check(name, tkgu, got, expected, what):
    log(f'{what} {name} {tkgu}: E-R {got:.2f} (expected {expected})')
    if abs(got - expected) > TOLERANCE:
        raise ValueError(f'{what} {name} {tkgu}: {got:.2f} != expected {expected}; not continuing')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out = {k: os.path.join(cfg['output_dir'], k) for k in
           ('er_column.json', 'er_column_cells.txt', 'deprecate_gold_not_in_snapshot.tsv')}
    for p in out.values():
        if os.path.exists(p):
            raise FileExistsError(p)
    os.makedirs(cfg['output_dir'], exist_ok=True)

    # 1. test set
    year_of, gold_test, emerging = {}, {}, {}
    for year in YEARS:
        files = sorted(glob.glob(os.path.join(cfg['test_set'], f'snapshot_{year}-01-01', '*.jsonl')))
        if not files:
            raise FileNotFoundError(f'no test files for {year} in {cfg["test_set"]}')
        for fp in files:
            with open(fp) as fh:
                for line in fh:
                    d = json.loads(line)
                    h = d['hash_id']
                    assert h not in year_of, f'duplicate hash_id {h}'
                    year_of[h] = year
                    for t in d['tkgu_triples']:
                        if 'd-triples' in t['tkgu_operations'] and supported(t, cfg['deprecate_assessor']):
                            gold_test.setdefault(h, []).append(tuple(t['triple']))
                            emerging[(h, tuple(t['triple']))] = t['emerging_head'] or t['emerging_tail']
    n_supported = sum(len(v) for v in gold_test.values())
    log(f'test set: {len(year_of)} instances; {n_supported} supported Deprecates in {len(gold_test)} instances')

    # 2. snapshots
    wanted = {y: set() for y in YEARS}
    for h, triples in gold_test.items():
        wanted[year_of[h]].update(triples)
    jobs = [(y, os.path.join(cfg['snapshots_dir'], f'kg_snapshot_{y}-01-01.tsv.gz'), wanted[y]) for y in YEARS]
    with Pool(len(jobs)) as pool:
        in_snapshot = dict(pool.map(find_in_snapshot, jobs))

    # 3. counts
    not_in = [(year_of[h], h, *t, 'minted_entity' if emerging[(h, t)] else 'entered_after_t')
              for h, triples in gold_test.items() for t in triples if t not in in_snapshot[year_of[h]]]
    n_minted = sum(1 for r in not_in if r[-1] == 'minted_entity')
    got = {'supported_deprecates': n_supported, 'in_snapshot': n_supported - len(not_in),
           'not_in_snapshot_minted_entity': n_minted, 'not_in_snapshot_existing_entities': len(not_in) - n_minted}
    log(f'counts: {got}')
    if got != cfg['expected_counts']:
        raise ValueError(f'counts {got} != expected_counts {cfg["expected_counts"]}; not continuing')

    # 4.-5. KG-aware rows
    results = {'counts': got, 'cells': {}}
    batch = load_batch(cfg['results_kg_aware'])
    for model, name in cfg['kg_aware_models'].items():
        cells = {}
        for tkgu in ('x-triples', 'e-triples', 'd-triples'):
            er, n = mean_recall(batch[model][tkgu], model, tkgu)
            check(name, tkgu, er, cfg['expected_s04_er'][name][tkgu], 's04 re-score')
            cells[tkgu] = {'all_supported_gold': er, 'instances_all': n, 'table3': er, 'instances_table3': n}
        rows = batch[model]['d-triples']
        pkl_gold = {r[0]: sorted(tuple(t) for t, ok in zip(r[1], r[3]) if ok) for r in rows}
        if {h: v for h, v in pkl_gold.items() if v} != {h: sorted(v) for h, v in gold_test.items()}:
            raise ValueError(f'{name}: supported Deprecate gold in the results file != test set')
        er, n = mean_recall(restrict_to_snapshot(rows, in_snapshot, year_of), model, 'd-triples')
        cells['d-triples'].update({'table3': er, 'instances_table3': n})
        results['cells'][name] = cells
        log(f'{name} Deprecate E-R: all supported gold {cells["d-triples"]["all_supported_gold"]:.2f} '
            f'({cells["d-triples"]["instances_all"]} instances) -> gold in T_t {er:.2f} ({n} instances)')
    del batch
    gc.collect()

    # 6. IE rows
    batch = load_batch(cfg['results_ie'])
    for model, name in cfg['ie_models'].items():
        results['cells'][name] = {}
        for tkgu, expected in cfg['expected_s04_er'][name].items():
            er, n = mean_recall(batch[model][tkgu], model, tkgu)
            check(name, tkgu, er, expected, 's04 re-score')
            results['cells'][name][tkgu] = {'all_supported_gold': er, 'instances_all': n,
                                            'table3': er, 'instances_table3': n}

    # 7. Table 3
    for name, expected in cfg['expected_table3_er'].items():
        for tkgu, value in expected.items():
            check(name, tkgu, results['cells'][name][tkgu]['table3'], value, 'Table 3')
    log('Table 3 E-R column: PASS')

    with open(out['er_column.json'], 'w') as f:
        json.dump(results, f, indent=2)
    with open(out['er_column_cells.txt'], 'w') as f:
        f.write('Table 3 E-R cells: row: Exists / Add / Deprecate\n')
        for name, cells in results['cells'].items():
            vals = [f'{cells[t]["table3"]:.1f}' if t in cells else '--' for t in ('x-triples', 'e-triples', 'd-triples')]
            f.write(f'{name}: {" / ".join(vals)}\n')
    with open(out['deprecate_gold_not_in_snapshot.tsv'], 'w') as f:
        f.write('year\thash_id\ts\tp\to\treason\n')
        for r in sorted(not_in):
            f.write('\t'.join(r) + '\n')
    log(f'saved {", ".join(out.values())}')
    print(open(out['er_column_cells.txt']).read(), flush=True)


if __name__ == '__main__':
    main()
