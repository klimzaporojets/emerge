"""scripts/kg_context/check_snapshot_consistency.py: the final check applies the construction's rules (tkgu_rules.py).

  - Deprecate has NO consistency rule: a text may deprecate a fact that is not in the snapshot, also one
    of an entity being minted (the paper's own example, (Adam Peters, member of, 49ers)). Requiring Deprecate triples
    to be in the snapshot was a bug that removed 144 correct gold triples.
  - Exists / Add triple rules must still fire.
  - "in KG" = in a snapshot triple OR in the entity dictionary: an existing entity found only in the dictionary is
    fine; an emerging entity found only in the dictionary is NOT new (merged Wikidata items).
  - the emerging flags must fit the op (Mint+Add >= 1 emerging, Infer exactly 1).
"""
import gzip
import importlib.util
import json
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _load():
    spec = importlib.util.spec_from_file_location(
        'check_snapshot_consistency', os.path.join(ROOT, 'scripts/kg_context/check_snapshot_consistency.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load()


def _triple(h, p, t, ops, eh=False, et=False):
    return {'triple': [h, p, t], 'tkgu_operations': ops, 'emerging_head': eh, 'emerging_tail': et,
            'triple_labels': [h, p, t]}


def _setup(tmp_path, triples, dictionary=()):
    snap = tmp_path / 'kg_snapshot_2019-01-01.tsv.gz'
    with gzip.open(snap, 'wt', encoding='utf-8') as f:
        f.write('Q1\tP31\tQ2\tone\tinstance of\ttwo\r\nQ3\tP99\tQ1\tthree\tp99\tone\r\n')
    dicts = tmp_path / 'relik_entity_index'
    (dicts / '2019-01-01').mkdir(parents=True)
    with gzip.open(dicts / '2019-01-01' / 'documents.jsonl.gz', 'wt', encoding='utf-8') as f:
        for i, q in enumerate(dictionary):
            f.write(json.dumps({'text': q, 'id': i, 'metadata': {'wikidata': q, 'definition': ''}}) + '\n')
    eval_dir = tmp_path / 'evaluation_set' / 'snapshot_2019-01-01'       # the released layout
    eval_dir.mkdir(parents=True)
    (eval_dir / 'delta_2019-01-08.jsonl').write_text(json.dumps({'hash_id': 'h1', 'tkgu_triples': triples}) + '\n')
    return str(snap), str(dicts), str(tmp_path / 'evaluation_set')


def _fails(tmp_path, triples, dictionary=()):
    snap, dicts, eval_dir = _setup(tmp_path, triples, dictionary)
    _, stats = C.audit_year(snap, dicts, eval_dir, '2019', max_ex=10)
    return stats


def test_year_files_are_matched_by_snapshot_folder_not_by_any_year_in_the_path(tmp_path):
    # a year inside the dataset path itself (here ".../2019_runs/...") must not pull in other years' files
    base = tmp_path / '2019_runs'
    base.mkdir()
    snap, dicts, eval_dir = _setup(base, [_triple('Q1', 'P31', 'Q2', ['x-triples'])])
    other = base / 'evaluation_set' / 'snapshot_2020-01-01'
    other.mkdir()
    (other / 'delta_2020-01-08.jsonl').write_text(json.dumps({'hash_id': 'h2', 'tkgu_triples': [
        _triple('Q7', 'P31', 'Q8', ['x-triples'])]}) + '\n')
    _, stats = C.audit_year(snap, dicts, eval_dir, '2019', max_ex=10)
    assert stats['gold'] == 1 and stats['fails'] == {}


def test_missing_year_and_malformed_triple_fail_loudly(tmp_path):
    snap, dicts, eval_dir = _setup(tmp_path, [{'triple': ['Q1', 'P31'], 'tkgu_operations': ['x-triples'],
                                               'emerging_head': False, 'emerging_tail': False}])
    with pytest.raises(ValueError):                                     # a triple without 3 elements
        C.audit_year(snap, dicts, eval_dir, '2019', max_ex=10)
    with pytest.raises(FileNotFoundError):                              # no files for the year
        C.audit_year(snap, dicts, eval_dir, '2020', max_ex=10)


def test_deprecate_needs_no_snapshot_presence(tmp_path):
    stats = _fails(tmp_path, [
        _triple('Q1', 'P31', 'Q2', ['x-triples']),                    # Exists, present        -> ok
        _triple('Q1', 'P31', 'Q2', ['d-triples']),                    # Deprecate, present     -> ok
        _triple('Q3', 'P17', 'Q2', ['d-triples']),                    # Deprecate, absent      -> ok
        _triple('Q777', 'P54', 'Q2', ['d-triples'], eh=True),         # Deprecate, minted head -> ok
        _triple('Q888', 'P54', 'Q999', ['d-triples']),                # Deprecate, unknown ids -> ok
    ])
    assert stats['fails'] == {}
    assert stats['op_counts']['deprecate']['n'] == 4


def test_exists_and_add_rules_still_fire(tmp_path):
    stats = _fails(tmp_path, [
        _triple('Q3', 'P17', 'Q1', ['x-triples']),                    # Exists, absent  -> TRIPLE-MISSING
        _triple('Q1', 'P31', 'Q2', ['e-triples']),                    # Add, present    -> TRIPLE-PRESENT
        _triple('Q3', 'P17', 'Q1', ['x-triples', 'd-triples']),      # not Deprecate-only -> still checked
    ])
    assert stats['fails'] == {'TRIPLE-MISSING': 2, 'TRIPLE-PRESENT': 1}
    # P17 is in no snapshot triple: reported as a note, never a failure (no TKGU rule on the relation; a property
    # can be new in Wikidata) -- the 5 corpus cases of 2026-10-27, e.g. (Europe PubMed Central, P7228, open access)
    assert stats['notes'] == {'RELATION-UNUSED': 2}


def test_entity_in_kg_means_triples_or_dictionary(tmp_path):
    stats = _fails(tmp_path, [
        _triple('Q8', 'P31', 'Q2', ['e-triples']),                    # Q8 only in the dictionary -> ok
        _triple('Q7', 'P31', 'Q2', ['e-triples']),                    # Q7 nowhere -> EXISTING-ENT-MISSING
        _triple('Q9', 'P31', 'Q2', ['ee-triples'], eh=True),          # "new" Q9 in the dictionary (merged item)
        _triple('Q60', 'P31', 'Q1', ['ee-kg-triples'], eh=True),      # really new -> ok
    ], dictionary=['Q8', 'Q9'])
    assert stats['fails'] == {'EXISTING-ENT-MISSING': 1, 'MINTED-ENT-PRESENT': 1}


def test_flags_must_fit_the_op(tmp_path):
    stats = _fails(tmp_path, [
        _triple('Q1', 'P31', 'Q3', ['ee-triples']),                   # Mint+Add without an emerging entity
        _triple('Q70', 'P31', 'Q71', ['ee-kg-triples'], eh=True, et=True),   # Infer with two emerging entities
        _triple('Q1', 'P31', 'Q3', ['x-triples', 'e-triples']),      # contradiction
    ])
    assert stats['fails'] == {'FLAG-STRUCTURE': 2, 'OP-CONTRADICTION': 1}
