"""src/dataset/emerge/tkgu_rules.py: the TKGU-operation rules against the KG at the snapshot date (manuscript §3).

The cases are the real failure modes found in the released v1.0.0 data:
  - a merged Wikidata item flagged as emerging (Mint+Add / Infer) although it is already in the KG
  - an Add whose triple is already in T_t -> relabelled as Exists (once; never duplicating an Exists)
  - an existing entity missing from the KG (in neither snapshot triples nor the entity dictionary)
  - an entity found ONLY in the entity dictionary counts as "in KG" (the agreed definition: triples OR dictionary)
  - Exists not in T_t; Mint+Add / Infer in T_t; wrong emerging flags
  - Deprecate has NO rule (requiring Deprecate triples to be in the snapshot wrongly removed 144 correct gold triples)
"""
import gzip
import json

import pytest

from dataset.emerge import tkgu_rules as R

K = ('Q1', 'P31', 'Q2')


def _decide(ops, in_t=False, eh=False, et=False, h_kg=True, t_kg=True, exists=None):
    return R.decide_ops(list(ops), K, eh, et, in_t, h_kg, t_kg, set() if exists is None else exists)


# ------------------------------------------------ rules ------------------------------------------------

def test_consistent_operations_are_kept():
    assert _decide(['x-triples'], in_t=True) == (['x-triples'], [])
    assert _decide(['e-triples']) == (['e-triples'], [])
    assert _decide(['ee-triples'], eh=True, h_kg=False) == (['ee-triples'], [])
    assert _decide(['ee-triples'], eh=True, et=True, h_kg=False, t_kg=False) == (['ee-triples'], [])
    assert _decide(['ee-kg-triples'], et=True, t_kg=False) == (['ee-kg-triples'], [])


def test_triple_rule():
    assert _decide(['x-triples'], in_t=False) == ([], [('x-triples', R.REMOVED, R.NOT_IN_T, None)])
    assert _decide(['ee-triples'], in_t=True, eh=True, h_kg=False) == \
        ([], [('ee-triples', R.REMOVED, R.IN_T, None)])                      # never relabelled
    assert _decide(['ee-kg-triples'], in_t=True, et=True, t_kg=False) == \
        ([], [('ee-kg-triples', R.REMOVED, R.IN_T, None)])


def test_merged_item_flagged_as_emerging_is_dropped():
    # the "new" head is a merged duplicate of an item already in the KG (found e.g. only in the dictionary)
    assert _decide(['ee-triples'], eh=True, h_kg=True) == ([], [('ee-triples', R.REMOVED, None, R.MINTED_IN_KG)])
    assert _decide(['ee-kg-triples'], et=True, t_kg=True) == \
        ([], [('ee-kg-triples', R.REMOVED, None, R.MINTED_IN_KG)])


def test_existing_entity_missing_from_the_kg_is_dropped():
    assert _decide(['e-triples'], t_kg=False) == ([], [('e-triples', R.REMOVED, None, R.EXISTING_NOT_IN_KG)])
    assert _decide(['x-triples'], in_t=False, h_kg=False) == \
        ([], [('x-triples', R.REMOVED, R.NOT_IN_T, R.EXISTING_NOT_IN_KG)])
    assert _decide(['ee-kg-triples'], eh=True, h_kg=False, t_kg=False) == \
        ([], [('ee-kg-triples', R.REMOVED, None, R.EXISTING_NOT_IN_KG)])


def test_flag_structure():
    assert _decide(['ee-triples']) == ([], [('ee-triples', R.REMOVED, None, R.FLAG_STRUCTURE)])         # none
    assert _decide(['ee-kg-triples'], eh=True, et=True, h_kg=False, t_kg=False) == \
        ([], [('ee-kg-triples', R.REMOVED, None, R.FLAG_STRUCTURE)])                                    # both
    assert _decide(['e-triples'], eh=True, h_kg=False) == ([], [('e-triples', R.REMOVED, None, R.FLAG_STRUCTURE)])
    assert _decide(['x-triples'], in_t=True, et=True) == ([], [('x-triples', R.REMOVED, None, R.FLAG_STRUCTURE)])


def test_add_in_t_becomes_exists_once():
    exists = set()
    assert _decide(['e-triples'], in_t=True, exists=exists) == \
        (['x-triples'], [('e-triples', R.RELABELLED_TO_EXISTS, R.IN_T, None)])
    assert exists == {K}
    # the same Add again in this instance -> only dropped (no duplicate Exists)
    assert _decide(['e-triples'], in_t=True, exists=exists) == \
        ([], [('e-triples', R.REMOVED_DUPLICATE_OF_EXISTS, R.IN_T, None)])


def test_add_in_t_with_a_broken_entity_rule_is_dropped_not_relabelled():
    assert _decide(['e-triples'], in_t=True, eh=True) == ([], [('e-triples', R.REMOVED, R.IN_T, R.FLAG_STRUCTURE)])


def test_op_order_is_kept_and_deprecate_is_untouched():
    assert _decide(['e-triples', 'd-triples'], in_t=True) == \
        (['x-triples', 'd-triples'], [('e-triples', R.RELABELLED_TO_EXISTS, R.IN_T, None)])
    for kw in (dict(), dict(in_t=True), dict(eh=True, et=True), dict(h_kg=False, t_kg=False), dict(eh=True)):
        assert _decide(['d-triples'], **kw) == (['d-triples'], [])


def test_unknown_operation_fails_loudly():
    with pytest.raises(ValueError):
        _decide(['z-triples'])


def test_entry_flags_must_be_bools():
    lookup = R.KgLookup({K}, {'Q1', 'Q2'}, set())
    with pytest.raises(TypeError):
        R.apply_rules_to_entry({'triple': list(K), 'tkgu_operations': ['x-triples'], 'emerging_head': 0,
                                'emerging_tail': False}, lookup, set())
    with pytest.raises(KeyError):
        R.apply_rules_to_entry({'triple': list(K), 'tkgu_operations': ['x-triples']}, lookup, set())


def _e(s, p, o, ops, eh=False, et=False):
    return {'triple': [s, p, o], 'tkgu_operations': ops, 'emerging_head': eh, 'emerging_tail': et}


def test_exists_of_instance_counts_only_exists_that_pass_the_rules():
    lookup = R.KgLookup({('Q1', 'P31', 'Q2'), ('Q5', 'P31', 'Q6')}, {'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'}, set())
    tkgu = [_e('Q1', 'P31', 'Q2', ['x-triples', 'd-triples']),   # valid Exists          -> counts
            _e('Q3', 'P17', 'Q4', ['x-triples']),                # not in T_t (dropped)  -> does not count
            _e('Q5', 'P31', 'Q6', ['x-triples'], eh=True),       # wrong flag (dropped)  -> does not count
            _e('Q7', 'P31', 'Q8', ['e-triples'])]
    assert R.exists_of_instance(tkgu, lookup) == {('Q1', 'P31', 'Q2')}


def test_add_is_kept_as_exists_when_the_same_exists_entry_is_dropped():
    # an Exists entry with a wrong emerging flag is dropped; the Add of the same triple (already in T_t, flags fine)
    # must then become the Exists instead of being dropped as its "duplicate" (the fact would otherwise vanish)
    lookup = R.KgLookup({('Q5', 'P31', 'Q6')}, {'Q5', 'Q6'}, set())
    tkgu = [_e('Q5', 'P31', 'Q6', ['x-triples'], eh=True), _e('Q5', 'P31', 'Q6', ['e-triples'])]
    exists = R.exists_of_instance(tkgu, lookup)
    kept = [R.apply_rules_to_entry(t, lookup, exists)[0] for t in tkgu]
    assert kept == [[], ['x-triples']]


# ---------------------------------------------- KG lookups ----------------------------------------------

def _kg(tmp_path, date, triples, dictionary):
    snaps, dicts = tmp_path / 'kg_snapshots', tmp_path / 'relik_entity_index'
    snaps.mkdir(exist_ok=True)
    (dicts / date).mkdir(parents=True, exist_ok=True)
    with gzip.open(snaps / f'kg_snapshot_{date}.tsv.gz', 'wt', encoding='utf-8') as f:
        for s, p, o in triples:
            f.write(f'{s}\t{p}\t{o}\tlabel {s}\tlabel {p}\tlabel {o}\r\n')     # released layout: 6 fields, CRLF
    with gzip.open(dicts / date / 'documents.jsonl.gz', 'wt', encoding='utf-8') as f:
        for i, q in enumerate(dictionary):
            f.write(json.dumps({'text': f'name {q}', 'id': i, 'metadata': {'wikidata': q, 'definition': 'x'}}) + '\n')
    return str(snaps), str(dicts)


def test_lookup_uses_triples_or_dictionary(tmp_path):
    snaps, dicts = _kg(tmp_path, '2019-01-01', [('Q1', 'P31', 'Q2'), ('Q3', 'P17', 'Q4')], ['Q1', 'Q9'])
    lookup = R.load_kg_lookup(snaps, dicts, '2019-01-01', {('Q1', 'P31', 'Q2'), ('Q1', 'P31', 'Q4')},
                              {'Q1', 'Q4', 'Q9', 'Q77'})
    assert lookup.triples_in_t == {('Q1', 'P31', 'Q2')}                # only wanted triples are kept
    assert lookup.entities_in_triples == {'Q1', 'Q4'}                  # subject or object; only wanted
    assert lookup.entities_in_dictionary == {'Q1', 'Q9'}
    assert lookup.in_kg('Q9') and lookup.in_kg('Q4') and not lookup.in_kg('Q77')
    assert lookup.in_t(('Q1', 'P31', 'Q2')) and not lookup.in_t(('Q1', 'P31', 'Q4'))


def test_lookups_per_date_and_missing_files_fail(tmp_path):
    snaps, dicts = _kg(tmp_path, '2019-01-01', [('Q1', 'P31', 'Q2')], ['Q1'])
    _kg(tmp_path, '2020-01-01', [], ['Q2'])
    lookups = R.load_kg_lookups(snaps, dicts, {'2019-01-01': ({('Q1', 'P31', 'Q2')}, {'Q1', 'Q2'}),
                                               '2020-01-01': ({('Q1', 'P31', 'Q2')}, {'Q1', 'Q2'})})
    assert lookups['2019-01-01'].in_t(('Q1', 'P31', 'Q2')) and not lookups['2020-01-01'].in_t(('Q1', 'P31', 'Q2'))
    assert not lookups['2020-01-01'].in_kg('Q1') and lookups['2020-01-01'].in_kg('Q2')
    with pytest.raises(FileNotFoundError):                              # missing snapshot
        R.load_kg_lookup(snaps, dicts, '2021-01-01', set(), set())
    (tmp_path / 'kg_snapshots' / 'kg_snapshot_2022-01-01.tsv.gz').write_bytes(
        (tmp_path / 'kg_snapshots' / 'kg_snapshot_2019-01-01.tsv.gz').read_bytes())
    with pytest.raises(FileNotFoundError):                              # missing dictionary
        R.load_kg_lookup(snaps, dicts, '2022-01-01', set(), set())


def test_malformed_kg_lines_fail(tmp_path):
    snaps, dicts = _kg(tmp_path, '2019-01-01', [], ['Q1'])
    with gzip.open(tmp_path / 'kg_snapshots' / 'kg_snapshot_2019-01-01.tsv.gz', 'wt', encoding='utf-8') as f:
        f.write('Q1\tP31\r\n')
    with pytest.raises(ValueError):
        R.load_kg_lookup(snaps, dicts, '2019-01-01', set(), {'Q1'})
    _kg(tmp_path, '2020-01-01', [('Q1', 'P31', 'Q2')], [])
    with gzip.open(tmp_path / 'relik_entity_index' / '2020-01-01' / 'documents.jsonl.gz', 'wt') as f:
        f.write(json.dumps({'text': 'no metadata'}) + '\n')
    with pytest.raises(KeyError):
        R.load_kg_lookup(snaps, dicts, '2020-01-01', set(), {'Q1'})
