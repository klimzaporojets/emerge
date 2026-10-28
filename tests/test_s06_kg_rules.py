"""s06_refactor_final_format: every TKGU operation must agree with the KG snapshot the instance refers to.

The released v1.0.1 data were corrected with the same rules (tkgu_rules.py), so construction and correction give the
same operations. Here: the snapshot date of an instance, the keys looked up in the KG (from all three triple lists),
and the per-instance enforcement (drop what breaks a rule; an Add already in T_t becomes an Exists, never a duplicate).
"""
import collections
import json

import pytest

from dataset.emerge import tkgu_rules as R
from dataset.emerge.s06_refactor_final_format import collect_lookup_keys, enforce_kg_rules, snapshot_date_of


def test_snapshot_date_of():
    assert snapshot_date_of({'delta_dates': ['2019-01-29', '2019-02-05']}) == '2019-01-01'
    assert snapshot_date_of({'interval_dates': ['2024-01-01', '2024-01-08']}) == '2024-01-01'   # before renaming
    for bad in (['2019-03-01', '2019-03-08'], ['19-01-01', '19-01-08']):
        with pytest.raises(ValueError):
            snapshot_date_of({'delta_dates': bad})
    with pytest.raises(KeyError):
        snapshot_date_of({})


def _raw(s, p, o, actions=('added_edge',), eh=False, et=False):
    return {'triple': [s, p, o], 'actions': list(actions), 'emerging_head': eh, 'emerging_tail': et}


def test_lookup_keys_come_from_all_three_lists_per_date(tmp_path):
    lines = [{'delta_dates': ['2019-01-08', '2019-01-15'],
              'emerging_knowledge': [_raw('Q1', 'P31', 'Q2'), _raw('Q9', 'P54', 'Q2', eh=True)],
              'existing_knowledge': [_raw('Q3', 'P17', 'Q4', actions=())],
              'matched_triples_entities_to_kg': [_raw('Q9', 'P27', 'Q5', eh=True)]},
             {'interval_dates': ['2020-01-01', '2020-01-08'],
              'emerging_knowledge': [_raw('Q1', 'P31', 'Q2', actions=('qualifier_added_edge',))],
              'existing_knowledge': [], 'matched_triples_entities_to_kg': []}]
    data = tmp_path / 'delta.jsonl'
    data.write_text(''.join(json.dumps(line) + '\n' for line in lines))
    assert collect_lookup_keys([data]) == {
        '2019-01-01': ({('Q1', 'P31', 'Q2'), ('Q9', 'P54', 'Q2'), ('Q3', 'P17', 'Q4'), ('Q9', 'P27', 'Q5')},
                       {'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q9'}),
        '2020-01-01': ({('Q1', 'P31', 'Q2')}, {'Q1', 'Q2'})}
    bad = tmp_path / 'bad.jsonl'
    bad.write_text(json.dumps({'delta_dates': ['2019-01-08', '2019-01-15'], 'emerging_knowledge': [],
                               'existing_knowledge': []}) + '\n')
    with pytest.raises(KeyError):                                       # a list missing -> fail loudly
        collect_lookup_keys([bad])


def _entry(s, p, o, ops, eh=False, et=False):
    return {'triple': [s, p, o], 'tkgu_operations': list(ops), 'emerging_head': eh, 'emerging_tail': et}


def test_enforce_kg_rules_on_one_instance():
    # KG at t: triples Q1-P31-Q2 and Q3-P17-Q4; entity Q8 only in the dictionary; Q50 a merged item already in the KG
    lookup = R.KgLookup({('Q1', 'P31', 'Q2'), ('Q3', 'P17', 'Q4')}, {'Q1', 'Q2', 'Q3', 'Q4', 'Q50'}, {'Q8'})
    tkgu = [_entry('Q1', 'P31', 'Q2', ['e-triples']),                 # Add in T_t            -> Exists
            _entry('Q1', 'P31', 'Q2', ['e-triples', 'd-triples']),    # same Add again        -> only Deprecate left
            _entry('Q3', 'P17', 'Q4', ['e-triples']),                 # Add in T_t, Exists below -> dropped
            _entry('Q8', 'P31', 'Q2', ['e-triples']),                 # Q8 in dictionary only -> kept (in KG)
            _entry('Q7', 'P31', 'Q2', ['e-triples']),                 # Q7 nowhere            -> dropped
            _entry('Q50', 'P54', 'Q2', ['ee-triples'], eh=True),      # merged item           -> dropped
            _entry('Q60', 'P54', 'Q2', ['ee-kg-triples'], eh=True),   # real new entity       -> kept
            _entry('Q3', 'P17', 'Q4', ['x-triples']),                 # Exists in T_t         -> kept
            _entry('Q3', 'P17', 'Q1', ['x-triples'])]                 # Exists not in T_t     -> dropped
    decisions = collections.Counter()
    kept = enforce_kg_rules(tkgu, lookup, decisions)
    assert [(t['triple'], t['tkgu_operations']) for t in kept] == [
        (['Q1', 'P31', 'Q2'], ['x-triples']), (['Q1', 'P31', 'Q2'], ['d-triples']),
        (['Q8', 'P31', 'Q2'], ['e-triples']), (['Q60', 'P54', 'Q2'], ['ee-kg-triples']),
        (['Q3', 'P17', 'Q4'], ['x-triples'])]
    assert decisions == collections.Counter({
        ('e-triples', R.RELABELLED_TO_EXISTS, R.IN_T, None): 1,
        ('e-triples', R.REMOVED_DUPLICATE_OF_EXISTS, R.IN_T, None): 2,
        ('e-triples', R.REMOVED, None, R.EXISTING_NOT_IN_KG): 1,
        ('ee-triples', R.REMOVED, None, R.MINTED_IN_KG): 1,
        ('x-triples', R.REMOVED, R.NOT_IN_T, None): 1})
