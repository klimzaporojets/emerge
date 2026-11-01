"""Deprecate Executable-R restricted to gold triples in the KG snapshot T_t
(scripts/stats/executable_recall_deprecate_in_snapshot.py).

Guards the camera-ready Table 3 rule: a supported gold Deprecate that is NOT in T_t (minted entity, or a
statement entered into Wikidata after t) cannot be executed on G_t, so it must neither count as a miss nor
as a hit. Cases: a not-in-T_t gold no longer lowers recall; an instance whose ONLY supported gold is not
in T_t is skipped (exclude_empty_gold, as in the s04 configs); a 405B-false gold stays excluded even if it
is in T_t; membership is per instance YEAR (a triple in the 2020 snapshot does not count for 2019).
Static, offline, <1 s.
"""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts/stats/executable_recall_deprecate_in_snapshot.py'


def _load():
    spec = importlib.util.spec_from_file_location('er_dep_in_snapshot', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()
A, B, C, D = ('Q1', 'P39', 'Q2'), ('Q3', 'P39', 'Q4'), ('Q5', 'P6', 'Q6'), ('Q7', 'P6', 'Q8')
YEAR_OF = {'h1': '2019', 'h2': '2019', 'h3': '2020'}
IN_SNAPSHOT = {'2019': {A, C}, '2020': {B}}


def _rows():
    return [
        ['h1', [list(A), list(B)], [list(A)], [True, True]],    # B supported but not in the 2019 snapshot
        ['h2', [list(D), list(C)], [list(C)], [True, False]],   # D not in T_t; C in T_t but 405B-false
        ['h3', [list(A), list(B)], [], [True, True]],           # A is in 2019, not in 2020 -> only B counts
    ]


def test_not_in_snapshot_gold_is_not_a_miss():
    rows = M.restrict_to_snapshot(_rows(), IN_SNAPSHOT, YEAR_OF)
    assert rows[0][3] == [True, False]
    er, n = M.mean_recall([rows[0]], 'm', 'd-triples')
    assert er == 100.0 and n == 1          # was 50.0 with B counted as a miss


def test_instance_without_executable_gold_is_skipped():
    rows = M.restrict_to_snapshot(_rows(), IN_SNAPSHOT, YEAR_OF)
    assert rows[1][3] == [False, False]    # 405B-false C stays excluded although it is in T_t
    _, n = M.mean_recall(rows, 'm', 'd-triples')
    assert n == 2                           # h2 dropped, h1 + h3 scored


def test_membership_is_per_instance_year():
    rows = M.restrict_to_snapshot(_rows(), IN_SNAPSHOT, YEAR_OF)
    assert rows[2][3] == [False, True]


def test_restriction_changes_the_mean_as_expected():
    er, n = M.mean_recall(_rows(), 'm', 'd-triples')
    assert n == 3 and abs(er - 100 * 0.5 / 3) < 1e-9     # all supported gold: h1 0.5, h2 0, h3 0
    er, n = M.mean_recall(M.restrict_to_snapshot(_rows(), IN_SNAPSHOT, YEAR_OF), 'm', 'd-triples')
    assert n == 2 and er == 50.0                          # gold in T_t: h1 1.0, h3 0 (h2 skipped)


def test_supported_flag_uses_first_matching_assessment():
    t = {'llm_assessment': [
        {'llm_name': 'X', 'llm_prompt_type': 'triple_deprecation', 'llm_assessment': True},
        {'llm_name': 'A', 'llm_prompt_type': 'triple_assertion', 'llm_assessment': True},
        {'llm_name': 'A', 'llm_prompt_type': 'triple_deprecation', 'llm_assessment': False},
        {'llm_name': 'A', 'llm_prompt_type': 'triple_deprecation', 'llm_assessment': True},
    ]}
    assert M.supported(t, 'A') is False
    assert M.supported({'llm_assessment': []}, 'A') is False
