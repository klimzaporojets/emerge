"""KG-aware few-shot builder must FAIL LOUDLY on a gold triple missing its
`emerging_head`/`emerging_tail` flag — never silently default to False.

A silent False would label an EMERGING entity as an existing one (wrong QID vs
MINTED in the few-shot answer), the exact silent-mislabel the repo's fail-hard rule
exists to prevent, and consistent with augment.gold_pieces which reads these flags
as required. Regression guard for scripts/kg_context/build_few_shot.py.

Imports the script module directly; skips cleanly if its deps aren't in the env.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _bfs():
    for p in (REPO / "src", REPO / "scripts" / "kg_context"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        import build_few_shot
        return build_few_shot
    except Exception as e:  # missing optional dep in a bare env -> don't fail the suite
        pytest.skip(f"build_few_shot not importable here: {e}")


# e-triples -> add -> "ADD"; emerging_tail=True -> tail rendered as MINTED.
VALID = {
    "triple": ["Q1", "P1", "Q2"],
    "triple_labels": ["Head", "rel", "Tail"],
    "tkgu_operations": ["e-triples"],
    "emerging_head": False,
    "emerging_tail": True,
}


def test_render_answer_valid_triple_renders():
    bfs = _bfs()
    out = bfs.render_answer([dict(VALID)])
    assert "ADD" in out, out            # op mapped through OP_OUT
    assert "MINTED" in out, out         # emerging_tail=True -> tail id is MINTED
    assert "Q1" in out, out             # non-emerging head keeps its QID


@pytest.mark.parametrize("missing", ["emerging_head", "emerging_tail"])
def test_render_answer_fails_hard_on_missing_emerging_flag(missing):
    bfs = _bfs()
    bad = {k: v for k, v in VALID.items() if k != missing}
    with pytest.raises(KeyError):
        bfs.render_answer([bad])
