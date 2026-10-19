"""Tests for the `exclude_empty_gold` toggle in cie_exact_match_scorer.

The scorer has no heavy deps (logging + typing), so this runs anywhere in <0.1s.
Guards the empty-gold E-R fix:
  - Option A (default, exclude_empty_gold=False): an instance whose 405B-filtered gold is EMPTY
    (a "Bucket B" instance) scores cie-recall 0 and is KEPT -> drags the mean (= published Table-3).
  - Option B (exclude_empty_gold=True): that instance is DROPPED entirely -> cie-recall consistent
    with completeness/graph-judge (which already exclude it, being per-triple).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.scorers.cie_exact_match_scorer import calculate_cie_exact_match  # noqa: E402


def _rows():
    # A: one gold triple, assessed TRUE, predicted exactly -> recall 1.0
    a = ['A', [['Q1', 'P1', 'Q2']], [['Q1', 'P1', 'Q2']], [True]]
    # B ("Bucket B"): one gold triple but assessed FALSE -> gt_set empty (nothing valid to recall)
    b = ['B', [['Q3', 'P3', 'Q4']], [['Q9', 'P9', 'Q9']], [False]]
    return [a, b]


def _recall_by_hash(res):
    return {r['hash_id']: r['score'] for r in res['scores_per_instance'] if r['metric'] == 'cie-recall'}


def test_option_a_keeps_empty_gold_as_zero_recall():
    rec = _recall_by_hash(calculate_cie_exact_match(_rows(), model='m', tkgu_type='e-triples',
                                                    exclude_empty_gold=False))
    assert rec == {'A': 1.0, 'B': 0.0}                 # Bucket B kept as 0
    assert sum(rec.values()) / len(rec) == 0.5          # mean dragged to 0.5


def test_option_b_excludes_empty_gold():
    rec = _recall_by_hash(calculate_cie_exact_match(_rows(), model='m', tkgu_type='e-triples',
                                                    exclude_empty_gold=True))
    assert rec == {'A': 1.0}                             # Bucket B dropped entirely
    assert sum(rec.values()) / len(rec) == 1.0          # mean over the valid population only


def test_default_is_option_a():
    # No flag -> must behave exactly like the published Table-3 scoring (Bucket B kept as 0).
    rec = _recall_by_hash(calculate_cie_exact_match(_rows(), model='m', tkgu_type='e-triples'))
    assert rec == {'A': 1.0, 'B': 0.0}
