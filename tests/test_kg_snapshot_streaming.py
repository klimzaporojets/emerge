"""Guards that KG snapshots are STREAMED from the shipped `.tsv.gz`, never decompressed to disk.

Failure modes this catches:
  * a reader using a plain `open()` on `kg_snapshot_<year>-01-01.tsv` -> breaks on the shipped gz-only
    data (and for the eval, silently degrades to an EMPTY snapshot -> relik-cie skipped);
  * `download_data.sh` regaining a `gunzip` step (~22 GB of decompressed TSVs on disk);
  * a stale or PARTIALLY-decompressed `.tsv` next to the `.gz` being read instead of the complete `.gz`;
  * the inlined resolver in the two standalone kg_context scripts drifting from src/kg_context/snapshot_io.py.

Pure-Python on tiny tmp files: runs anywhere in well under a second.
"""
import gzip
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kg_context.snapshot_io import open_kg_snapshot, resolve_kg_snapshot  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LINES = ['Q1\tP31\tQ5\tAlice\tinstance of\thuman\n', 'Q2\tP54\tQ9\tBob\tmember of\tTeam\n']


def _write_gz(path, lines):
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        f.writelines(lines)


def _write_plain(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def _load_script(relpath, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------- resolver ----------------

def test_resolve_prefers_gz_over_plain(tmp_path):
    """Both present (e.g. a half-finished `gunzip -k`): the complete .gz must win."""
    tsv = tmp_path / 'kg_snapshot_2019-01-01.tsv'
    _write_gz(str(tsv) + '.gz', LINES)
    _write_plain(tsv, LINES[:1])  # partial decompress
    assert resolve_kg_snapshot(str(tsv)) == str(tsv) + '.gz'


def test_resolve_falls_back_to_plain_tsv(tmp_path):
    tsv = tmp_path / 'kg_snapshot_2019-01-01.tsv'
    _write_plain(tsv, LINES)
    assert resolve_kg_snapshot(str(tsv)) == str(tsv)


def test_resolve_explicit_gz_path(tmp_path):
    gz = tmp_path / 'kg_snapshot_2019-01-01.tsv.gz'
    _write_gz(gz, LINES)
    assert resolve_kg_snapshot(str(gz)) == str(gz)


def test_resolve_missing_returns_none(tmp_path):
    assert resolve_kg_snapshot(str(tmp_path / 'kg_snapshot_2019-01-01.tsv')) is None
    # an explicitly requested .gz must NOT silently fall back to a plain .tsv
    _write_plain(tmp_path / 'kg_snapshot_2020-01-01.tsv', LINES)
    assert resolve_kg_snapshot(str(tmp_path / 'kg_snapshot_2020-01-01.tsv.gz')) is None


# ---------------- opener ----------------

def test_open_streams_gz_without_writing_to_disk(tmp_path):
    """Configs name the plain `.tsv`; only the `.gz` exists -> read it, and leave NO .tsv behind."""
    tsv = tmp_path / 'kg_snapshot_2019-01-01.tsv'
    _write_gz(str(tsv) + '.gz', LINES)
    with open_kg_snapshot(str(tsv)) as f:
        assert f.readlines() == LINES
    assert sorted(os.listdir(tmp_path)) == ['kg_snapshot_2019-01-01.tsv.gz']


def test_gz_and_plain_parse_to_identical_triples(tmp_path):
    """Streaming the .gz must give exactly the triples the old plain-TSV read gave."""
    gz_dir, tsv_dir = tmp_path / 'gz', tmp_path / 'tsv'
    gz_dir.mkdir()
    tsv_dir.mkdir()
    _write_gz(gz_dir / 'kg.tsv.gz', LINES)
    _write_plain(tsv_dir / 'kg.tsv', LINES)

    def triples(p):
        with open_kg_snapshot(str(p)) as f:
            return {tuple(line.strip().split('\t', 3)[:3]) for line in f}

    assert triples(gz_dir / 'kg.tsv') == triples(tsv_dir / 'kg.tsv') == {('Q1', 'P31', 'Q5'), ('Q2', 'P54', 'Q9')}


def test_open_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_kg_snapshot(str(tmp_path / 'kg_snapshot_2019-01-01.tsv'))


# ---------------- standalone scripts stay in sync ----------------

@pytest.mark.parametrize('relpath,name', [
    ('scripts/kg_context/check_snapshot_consistency.py', 'check_snapshot_consistency'),
    ('scripts/kg_context/filter_indices_to_snapshot.py', 'filter_indices_to_snapshot'),
])
def test_standalone_resolver_matches_shared_helper(tmp_path, relpath, name):
    resolve_snapshot = _load_script(relpath, name).resolve_snapshot
    both = tmp_path / 'both.tsv'
    _write_gz(str(both) + '.gz', LINES)
    _write_plain(both, LINES[:1])
    plain = tmp_path / 'plain.tsv'
    _write_plain(plain, LINES)
    for p in (both, plain, tmp_path / 'missing.tsv', tmp_path / 'missing.tsv.gz'):
        assert resolve_snapshot(str(p)) == resolve_kg_snapshot(str(p))


# ---------------- no decompression anywhere ----------------

def test_download_script_does_not_decompress():
    with open(os.path.join(ROOT, 'scripts', 'download_data.sh'), encoding='utf-8') as f:
        src = f.read()
    assert 'gunzip' not in src, 'download_data.sh must keep KG snapshots gzipped (readers stream them)'


@pytest.mark.parametrize('relpath', [
    'src/evaluation/s04_evaluate_predictions.py',
    'src/kg_context/kg_lookup.py',
    'src/kg_context/index_builder.py',
])
def test_src_snapshot_readers_use_streaming_helper(relpath):
    with open(os.path.join(ROOT, relpath), encoding='utf-8') as f:
        src = f.read()
    assert 'open_kg_snapshot' in src, f'{relpath} must read KG snapshots via kg_context.snapshot_io.open_kg_snapshot'
