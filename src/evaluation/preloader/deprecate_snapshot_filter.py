"""A2 deprecate-precondition filter — pure helpers (no heavy deps, unit-testable anywhere).

A DEPRECATE (d-triples) gold triple must satisfy the precondition (s,p,o) in T_t: the FULL
triple must be present in the current KG snapshot T_t (you cannot deprecate an edge that is not
there). Triples that violate this are mislabeled and are dropped when the evaluation runs with
ArgsEvaluation.exclude_deprecate_not_in_snapshot=True (see preloader_emerge).

IMPORTANT — this checks the FULL triple, not just the subject. Verified on all 7 snapshots:
  * full-triple (s,p,o) not in T_t : 144 / 1628  -> corrected denominator 1484   [THE correct check]
  * subject not in V_t             :  46 / 1628                                   (too weak)
  * control: Exists gold full-triple IN T_t = 99.3% -> the snapshot IS triple-complete, so a
    missing full triple is a real precondition violation, not a snapshot-coverage artifact.
Kept free of pandas/torch so the logic can be tested in <0.1s in any environment.
"""


def snapshot_triple_set(snapshot_to_triples, str_year):
    """The KG snapshot triple set T_t = {(h, r, t)} for `str_year`.

    Returns None when that snapshot is unavailable or empty, which makes the deprecate filter
    a NO-OP for that year (a missing snapshot must never drop all deprecate gold).
    Optionality is made explicit (no dict.get-with-default)."""
    if snapshot_to_triples is None or str_year not in snapshot_to_triples:
        return None
    triples = snapshot_to_triples[str_year]
    if not triples:
        return None
    return triples


def deprecate_gold_out_of_snapshot(triple, snapshot_triples):
    """True iff a DEPRECATE gold triple (h, r, t) is absent from T_t, violating (s,p,o) in T_t.

    `triple` is (h, r, t) (list or tuple); `snapshot_triples` is a set of (h, r, t) tuples, or
    None when T_t is unavailable -> returns False (no-op)."""
    if snapshot_triples is None:
        return False
    return tuple(triple) not in snapshot_triples
