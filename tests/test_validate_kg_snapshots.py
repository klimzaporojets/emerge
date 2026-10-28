"""scripts/kg_context/validate_kg_snapshots.py: every check must catch the corruption it is meant for.

Real failure modes behind the checks: a truncated download (F1), a label with an embedded TAB/newline that splits
a triple across fields/lines and is then silently skipped by the readers (F4), an empty label instead of the id
fallback (F6, the old emerge-prod behaviour), labels from two different label files (F7/X1), metadata or checksum
not matching the files (F9/F10).
"""
import gzip
import hashlib
import importlib.util
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _load():
    name = 'validate_kg_snapshots'
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'scripts/kg_context/validate_kg_snapshots.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # registered so the worker pool can pickle its functions
    spec.loader.exec_module(mod)
    return mod


V = _load()
GOOD = ['Q1\tP31\tQ2\tÄrzte ✓\tinstance of\ttwo', 'Q2\tP17\tQ3\ttwo\tcountry\tthree',
        'Q3\tP999\tQ1\tthree\tP999\tÄrzte ✓']


def _write(path, lines, terminator='\r\n', raw=None):
    with gzip.open(path, 'wb') as f:
        f.write(raw if raw is not None else ''.join(l + terminator for l in lines).encode('utf-8'))
    return str(path)


def _checks(path):
    return set(V.validate_file(path)['failures'])


def test_valid_file_passes(tmp_path):
    res = V.validate_file(_write(tmp_path / 'kg_snapshot_2019-01-01.tsv.gz', GOOD))
    assert res['failures'] == {}
    assert res['n_lines'] == 3 and len(res['entity_label']) == 3 and len(res['relation_label']) == 3


def test_each_corruption_is_caught(tmp_path):
    cases = {
        'F4 6 fields': GOOD[:1] + ['Q2\tP17\tQ3\ttwo\tcountry'],                       # field lost
        'F4 empty line': GOOD[:1] + [''] + GOOD[1:],
        'F5 id format': GOOD[:1] + ['Q2\tP17\tX3\ttwo\tcountry\tthree'],
        'F6 empty label': GOOD[:1] + ['Q2\tP17\tQ3\t\tcountry\tthree'],
        'F7 one label per id': GOOD + ['Q1\tP17\tQ3\tother label\tcountry\tthree'],
        'F8 no duplicate triple': GOOD + GOOD[:1],
    }
    for i, (check, lines) in enumerate(cases.items()):
        assert check in _checks(_write(tmp_path / f'c{i}.tsv.gz', lines)), check
    # a label with an embedded newline splits one triple over two lines
    assert 'F4 6 fields' in _checks(_write(tmp_path / 'nl.tsv.gz', ['Q1\tP31\tQ2\tline\nbreak\tinstance of\ttwo']))


def test_encoding_terminator_and_truncation(tmp_path):
    good_bytes = ''.join(l + '\r\n' for l in GOOD).encode('utf-8')
    assert 'F2 utf-8' in _checks(_write(tmp_path / 'u.tsv.gz', None, raw=good_bytes + b'Q4\tP1\tQ5\t\xff\tx\ty\r\n'))
    assert 'F3 one line terminator' in _checks(_write(tmp_path / 't.tsv.gz', None, raw=good_bytes + b'Q4\tP1\tQ5\ta\tb\tc\n'))
    full = tmp_path / 'full.tsv.gz'
    _write(full, GOOD * 50)
    truncated = tmp_path / 'trunc.tsv.gz'
    truncated.write_bytes(full.read_bytes()[:-20])
    assert 'F1 gzip complete' in _checks(str(truncated))


def _release(tmp_path, name, files):
    d = tmp_path / name
    d.mkdir()
    for date, lines in files.items():
        _write(d / f'kg_snapshot_{date}.tsv.gz', lines)
    return d


def test_run_metadata_sha_reference_and_cross_file(tmp_path, capsys):
    new = _release(tmp_path, 'new', {'2019-01-01': GOOD[:2], '2020-01-01': GOOD})
    old = _release(tmp_path, 'old', {'2019-01-01': GOOD[:1], '2020-01-01': GOOD})
    meta = tmp_path / 'meta'
    meta.mkdir()
    for date, n in (('2019-01-01', 2), ('2020-01-01', 3)):
        (meta / f'metadata_{date}.json').write_text(json.dumps({'nr_active_triples': n}))
    sums = tmp_path / 'SHA256SUMS'
    sums.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n'
                            for p in sorted(new.iterdir())))
    assert V.run(str(new), str(meta), str(sums), str(old), workers=2) == 0
    out = capsys.readouterr().out
    assert 'VERDICT: PASS' in out
    assert 'INFO X3 2019-01-01: reference=1 validated=2 only_in_reference=0 only_in_validated=1' in out

    (meta / 'metadata_2019-01-01.json').write_text(json.dumps({'nr_active_triples': 5}))
    assert V.run(str(new), str(meta), None, None, workers=1) == 1
    assert 'F9 metadata count' in capsys.readouterr().out

    sums.write_text(sums.read_text().replace(sums.read_text()[:8], '00000000', 1))
    assert V.run(str(new), None, str(sums), None, workers=1) == 1
    assert 'F10 sha256' in capsys.readouterr().out


def test_label_conflict_across_files(tmp_path, capsys):
    d = _release(tmp_path, 'x', {'2019-01-01': GOOD, '2020-01-01': ['Q1\tP31\tQ2\trenamed\tinstance of\ttwo']})
    assert V.run(str(d), None, None, None, workers=1) == 1
    assert 'FAIL X1 entity_label identical across snapshots' in capsys.readouterr().out
