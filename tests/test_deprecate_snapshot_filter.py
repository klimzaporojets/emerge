"""Tests for the A2 deprecate-precondition filter helpers in deprecate_snapshot_filter.

Guards the fix in rebuttal/neurips_2026/BUG_deprecate_not_in_snapshot_gold.md:
a DEPRECATE gold triple (h,r,t) that is absent from the KG snapshot T_t violates the
precondition (s,p,o) in T_t and must be dropped when exclude_deprecate_not_in_snapshot=True.

CRUCIAL: the check is on the FULL triple, not just the subject. Verified on all 7 snapshots
(writeup/iclr_2027/iclr_final_camera_ready_neurips.md): full-triple ∉ T_t = 144, subject ∉ V_t
= 46; the extra 98 have a present subject but an absent edge, and must also be dropped.

The helpers are pure functions, so this runs anywhere in <0.1s.
SAFETY: a missing/empty snapshot must be a NO-OP (return None / False) — never drop all gold.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.preloader.deprecate_snapshot_filter import (  # noqa: E402
    snapshot_triple_set,
    deprecate_gold_out_of_snapshot,
)

# snapshot T_t as stored by the pipeline: {year: set of (h, r, t)}
SNAP = {'2020': {('Q1', 'P1', 'Q2'), ('Q3', 'P2', 'Q1')}}


def test_snapshot_triple_set_returns_the_year_set():
    assert snapshot_triple_set(SNAP, '2020') == {('Q1', 'P1', 'Q2'), ('Q3', 'P2', 'Q1')}


def test_snapshot_triple_set_missing_year_is_none():
    assert snapshot_triple_set(SNAP, '2019') is None


def test_snapshot_triple_set_none_container_is_none():
    assert snapshot_triple_set(None, '2020') is None


def test_snapshot_triple_set_empty_is_none():
    assert snapshot_triple_set({'2020': set()}, '2020') is None


def test_full_triple_present_is_kept():
    snap = snapshot_triple_set(SNAP, '2020')
    assert deprecate_gold_out_of_snapshot(['Q1', 'P1', 'Q2'], snap) is False  # (s,p,o) in T_t -> keep


def test_full_triple_absent_is_dropped():
    snap = snapshot_triple_set(SNAP, '2020')
    assert deprecate_gold_out_of_snapshot(['Q9', 'P9', 'Q9'], snap) is True  # not in T_t -> drop


def test_present_subject_but_absent_edge_is_dropped():
    # THE distinguishing case vs a subject-only check: subject Q1 IS in T_t, but this edge is not.
    # Full-triple check must still drop it (this is the +98 beyond the 46 subject-missing).
    snap = snapshot_triple_set(SNAP, '2020')
    assert deprecate_gold_out_of_snapshot(['Q1', 'P1', 'Q999'], snap) is True


def test_tuple_and_list_triples_are_equivalent():
    snap = snapshot_triple_set(SNAP, '2020')
    assert deprecate_gold_out_of_snapshot(('Q1', 'P1', 'Q2'), snap) is False


def test_missing_snapshot_is_noop_and_keeps_gold():
    # snapshot None (unavailable) -> never drop: the guardrail against wiping all gold.
    assert deprecate_gold_out_of_snapshot(['Q9', 'P9', 'Q9'], None) is False
