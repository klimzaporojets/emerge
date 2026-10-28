#!/usr/bin/env python3
"""Validate released KG snapshot files `kg_snapshot_<date>.tsv.gz` (read-only; streams the .gz, never decompresses).

Every consumer in this repo reads a snapshot line by line, strips the line end and splits on TAB
(`kg_lookup`, `index_builder`, `s04_evaluate_predictions`). A line that does not split into the expected
fields is silently skipped by some of them, so the format must hold on EVERY line. Checks per file:

  F1 gzip stream complete and CRC-valid (a truncated download fails here)
  F2 UTF-8 decodable
  F3 one line terminator style for the whole file (the released files use CRLF from csv.writer)
  F4 no empty line; exactly 6 TAB-separated fields per line
  F5 subject/object match Q<digits>, relation matches P<digits>
  F6 no empty label (a missing label is written as the id itself)
  F7 one label per entity id and per relation id within the file
  F8 no duplicate (subject, relation, object)
  F9 (--metadata-dir) number of lines == metadata `nr_active_triples`
  F10 (--sha256sums) SHA-256 of the .gz == the SHA256SUMS entry
Across files:
  X1 entity / relation labels identical across all snapshots (they come from one label file)
  X2 snapshots grow over time (WARN only: deletions can in principle outweigh additions)
  X3 (--reference-dir) triples only in the reference vs only in the validated snapshot, per date (INFO)

Exit code 1 if any F* or X1 check fails.

USAGE:
  python scripts/kg_context/validate_kg_snapshots.py --snapshots-dir DIR \
      [--metadata-dir DIR] [--sha256sums FILE] [--reference-dir OLD_DIR] [--workers 3]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

QID_RE = re.compile(r'Q[1-9][0-9]*\Z')
PID_RE = re.compile(r'P[1-9][0-9]*\Z')
FILE_RE = re.compile(r'kg_snapshot_(\d{4}-\d{2}-\d{2})\.tsv\.gz\Z')
PROGRESS_SECS = 30
MAX_EXAMPLES = 5


def log(msg: str) -> None:
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}', flush=True)


def triple_key(s: str, p: str, o: str) -> int:
    """Stable 64-bit key of a triple (same in every process, unlike hash())."""
    return int.from_bytes(hashlib.blake2b(f'{s}\t{p}\t{o}'.encode(), digest_size=8).digest(), 'little')


def label_key(label: str) -> int:
    return zlib.crc32(label.encode())


def find_snapshots(snapshots_dir: str) -> dict[str, str]:
    found = {}
    for path in sorted(glob.glob(os.path.join(snapshots_dir, 'kg_snapshot_*.tsv.gz'))):
        m = FILE_RE.search(os.path.basename(path))
        if m:
            found[m.group(1)] = path
    if not found:
        raise SystemExit(f'no kg_snapshot_<date>.tsv.gz in {snapshots_dir}')
    return found


def validate_file(path: str, tag: str = '') -> dict:
    """Stream one snapshot and return counts, failure examples and the data needed for cross-file checks."""
    failures: dict[str, list[str]] = {}   # check -> first MAX_EXAMPLES examples
    fail_counts: dict[str, int] = {}      # check -> number of failing lines

    def fail(check: str, example: str) -> None:
        if check not in failures:
            failures[check] = []
            fail_counts[check] = 0
        if len(failures[check]) < MAX_EXAMPLES:
            failures[check].append(example)
        fail_counts[check] += 1

    entity_label: dict[str, int] = {}
    relation_label: dict[str, int] = {}
    keys = []
    terminators: dict[str, int] = {}
    n_lines = 0
    start = last = time.time()
    try:
        with gzip.open(path, 'rb') as f:
            for n_lines, raw in enumerate(f, start=1):
                if time.time() - last >= PROGRESS_SECS:
                    log(f'{tag} {n_lines:,} lines ({time.time() - start:.0f}s)')
                    last = time.time()
                term = '\r\n' if raw.endswith(b'\r\n') else ('\n' if raw.endswith(b'\n') else 'none')
                terminators[term] = terminators[term] + 1 if term in terminators else 1
                try:
                    line = raw.decode('utf-8')
                except UnicodeDecodeError as e:
                    fail('F2 utf-8', f'line {n_lines}: {e}')
                    continue
                line = line.rstrip('\n').rstrip('\r')
                if not line:
                    fail('F4 empty line', f'line {n_lines}')
                    continue
                p = line.split('\t')
                if len(p) != 6:
                    fail('F4 6 fields', f'line {n_lines}: {len(p)} fields: {line[:120]!r}')
                    continue
                s, r, o, sl, rl, ol = p
                if not QID_RE.match(s) or not QID_RE.match(o) or not PID_RE.match(r):
                    fail('F5 id format', f'line {n_lines}: {s} {r} {o}')
                if not sl or not rl or not ol:
                    fail('F6 empty label', f'line {n_lines}: {line[:120]!r}')
                for ident, lab, table in ((s, sl, entity_label), (o, ol, entity_label), (r, rl, relation_label)):
                    k = label_key(lab)
                    if ident in table:
                        if table[ident] != k:
                            fail('F7 one label per id', f'line {n_lines}: {ident} has two labels (e.g. {lab!r})')
                    else:
                        table[ident] = k
                keys.append(triple_key(s, r, o))
    except (EOFError, OSError, zlib.error) as e:
        fail('F1 gzip complete', f'after {n_lines} lines: {e}')

    if len(terminators) > 1:
        fail('F3 one line terminator', f'mixed terminators: {terminators}')
    key_arr = np.unique(np.array(keys, dtype=np.uint64))
    n_dup = len(keys) - len(key_arr)
    if n_dup:
        fail('F8 no duplicate triple', f'{n_dup:,} duplicate (s,p,o) lines')
    log(f'{tag} done: {n_lines:,} lines, {len(entity_label):,} entities, {len(relation_label):,} relations '
        f'({time.time() - start:.0f}s)')
    return {'n_lines': n_lines, 'terminators': terminators, 'failures': failures, 'fail_counts': fail_counts,
            'keys': key_arr, 'entity_label': entity_label, 'relation_label': relation_label}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def read_sha256sums(path: str) -> dict[str, str]:
    sums = {}
    with open(path, 'rt') as f:
        for line in f:
            if line.strip():
                digest, name = line.split(maxsplit=1)
                sums[os.path.basename(name.strip().lstrip('*'))] = digest
    return sums


def metadata_count(metadata_dir: str, date: str) -> int:
    path = os.path.join(metadata_dir, f'metadata_{date}.json')
    with open(path, 'rt') as f:
        return json.load(f)['nr_active_triples']


def _validate_job(args):
    return args[0], validate_file(args[1], tag=args[0])


def run(snapshots_dir: str, metadata_dir: str | None, sha256sums: str | None, reference_dir: str | None,
        workers: int) -> int:
    snaps = find_snapshots(snapshots_dir)
    log(f'validating {len(snaps)} snapshots in {snapshots_dir}: {", ".join(snaps)}')
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = dict(ex.map(_validate_job, list(snaps.items())))

    if metadata_dir is not None:
        for date, res in results.items():
            expected = metadata_count(metadata_dir, date)
            if expected != res['n_lines']:
                res['failures'].setdefault('F9 metadata count', []).append(
                    f'metadata nr_active_triples={expected:,} but {res["n_lines"]:,} lines')
    if sha256sums is not None:
        sums = read_sha256sums(sha256sums)
        for date, path in snaps.items():
            name = os.path.basename(path)
            if name not in sums:
                results[date]['failures'].setdefault('F10 sha256', []).append(f'{name} not in {sha256sums}')
            elif sha256_of(path) != sums[name]:
                results[date]['failures'].setdefault('F10 sha256', []).append(f'{name}: checksum mismatch')

    n_fail = 0
    print('\n=== per-file checks ===', flush=True)
    for date, res in results.items():
        status = 'FAIL' if res['failures'] else 'PASS'
        n_fail += bool(res['failures'])
        print(f'{status} {date}: {res["n_lines"]:,} lines, {len(res["entity_label"]):,} entities, '
              f'{len(res["relation_label"]):,} relations, terminators={res["terminators"]}', flush=True)
        for check, examples in res['failures'].items():
            count = res['fail_counts'][check] if check in res['fail_counts'] else len(examples)
            print(f'   {check}: {count:,} x, e.g. {examples}', flush=True)

    print('\n=== cross-file checks ===', flush=True)
    dates = sorted(results)
    for kind in ('entity_label', 'relation_label'):
        merged: dict[str, int] = {}
        conflicts = []
        for date in dates:
            for ident, k in results[date][kind].items():
                if ident in merged:
                    if merged[ident] != k and len(conflicts) < MAX_EXAMPLES:
                        conflicts.append(f'{ident} ({date})')
                else:
                    merged[ident] = k
        if conflicts:
            n_fail += 1
            print(f'FAIL X1 {kind} identical across snapshots: e.g. {conflicts}', flush=True)
        else:
            print(f'PASS X1 {kind} identical across snapshots ({len(merged):,} ids)', flush=True)
    sizes = [results[d]['n_lines'] for d in dates]
    grows = all(a <= b for a, b in zip(sizes, sizes[1:]))
    print(f'{"PASS" if grows else "WARN"} X2 snapshots grow over time: '
          + ', '.join(f'{d}={n:,}' for d, n in zip(dates, sizes)), flush=True)

    if reference_dir is not None:
        ref = find_snapshots(reference_dir)
        print(f'\n=== X3 vs reference {reference_dir} (INFO) ===', flush=True)
        for date in dates:
            if date not in ref:
                print(f'INFO X3 {date}: no reference snapshot', flush=True)
                continue
            ref_res = validate_file(ref[date], tag=f'reference {date}')
            only_ref = len(np.setdiff1d(ref_res['keys'], results[date]['keys'], assume_unique=True))
            only_new = len(np.setdiff1d(results[date]['keys'], ref_res['keys'], assume_unique=True))
            print(f'INFO X3 {date}: reference={ref_res["n_lines"]:,} validated={results[date]["n_lines"]:,} '
                  f'only_in_reference={only_ref:,} only_in_validated={only_new:,}', flush=True)

    print(f'\nVERDICT: {"FAIL" if n_fail else "PASS"} ({n_fail} failing file/cross checks)', flush=True)
    return 1 if n_fail else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snapshots-dir', required=True)
    ap.add_argument('--metadata-dir', default=None, help='dir with metadata_<date>.json (nr_active_triples)')
    ap.add_argument('--sha256sums', default=None, help='SHA256SUMS file listing the .gz files')
    ap.add_argument('--reference-dir', default=None, help='older snapshots to compare against (INFO only)')
    ap.add_argument('--workers', type=int, default=3, help='files validated in parallel (~3 GB RAM each)')
    a = ap.parse_args()
    sys.exit(run(a.snapshots_dir, a.metadata_dir, a.sha256sums, a.reference_dir, a.workers))
