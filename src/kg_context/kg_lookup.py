"""KGLookup: exact, structured queries over the KG snapshot (loaded whole into RAM).

NOT an embedding index ("index" = the dense embedding matrices in index_builder.py /
retriever.py). This is the opposite: EXACT graph queries. Three kinds:

  has_triple(h, r, t)                          -> does this EXACT triple exist?
                                                  (EXISTS/DEPRECATE: check a specific edge)
  edges_of(anchor, direction="out|in|both")    -> the anchor's edges, by direction
  existing_among(entities, relations, anchor)  -> existing triples AMONG the retrieved
                                                  entities/relations (BOTH directions),
                                                  for the KG-RAG context block

Indexes BOTH directions: out-adjacency (head -> {(pid, tail)}) and in-adjacency
(tail -> {(pid, head)}). `has_triple` and `existing_among` (ordered pairs over
out-adjacency) already cover incoming edges; in-adjacency is there for `edges_of`.

Loaded WHOLE by default (~6-8 GB / ~1-2 min for a full yearly snapshot) so
`existing_among` works over index-wide retrieved entities, not just the anchor's
neighborhood. Pass `target_anchors` to keep only triples TOUCHING those anchors
(head OR tail) -> fast smoke tests / anchor-only scope.

Snapshot line = `head_qid \t rel_pid \t tail_qid [\t h_label \t r_label \t t_label]`.

CLI smoke test:
    PYTHONPATH=src python -m kg_context.kg_lookup \
        --snapshot .../kg_snapshot_2019-01-01_top_1000.tsv --anchor Q1000001 --whole
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from kg_context.snapshot_io import open_kg_snapshot


class KGLookup:
    def __init__(self) -> None:
        self.out_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)  # head -> {(pid, tail)}
        self.in_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)   # tail -> {(pid, head)}
        self.labels: dict[str, str] = {}
        self.n_triples = 0

    @classmethod
    def load(
        cls,
        snapshot_path: str,
        target_anchors: set[str] | None = None,
        load_labels: bool = True,
    ) -> "KGLookup":
        import os as _os
        from tqdm import tqdm

        kg = cls()
        keep = target_anchors  # None => whole snapshot; else keep triples touching an anchor
        desc = f"KG-LOOKUP scan {_os.path.basename(snapshot_path)}"
        with open_kg_snapshot(snapshot_path) as f:  # streams the .gz; never decompress
            for raw in tqdm(f, desc=desc, unit="ln", unit_scale=True, mininterval=1.0):
                line = raw.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                p = line.split("\t")
                if len(p) < 3:
                    continue
                h, r, t = p[0], p[1], p[2]
                if keep is not None and h not in keep and t not in keep:
                    continue
                kg.out_edges[h].add((r, t))
                kg.in_edges[t].add((r, h))
                kg.n_triples += 1
                if load_labels and len(p) >= 6:
                    kg.labels.setdefault(h, p[3])
                    kg.labels.setdefault(r, p[4])
                    kg.labels.setdefault(t, p[5])
        return kg

    def has_triple(self, head_qid: str, pid: str, tail_qid: str) -> bool:
        """Exact whole-triple membership (direction-agnostic: pass head/tail in order)."""
        return (pid, tail_qid) in self.out_edges.get(head_qid, ())

    def label(self, ident: str) -> str:
        return self.labels.get(ident, ident)

    def edges_of(self, anchor_qid: str, direction: str = "both") -> list[tuple[str, str, str]]:
        """The anchor's edges as (head, rel, tail) QID triples, filtered by direction:
        'out' (anchor is head), 'in' (anchor is tail), 'both'."""
        res: list[tuple[str, str, str]] = []
        if direction in ("out", "both"):
            for (r, t) in self.out_edges.get(anchor_qid, ()):
                res.append((anchor_qid, r, t))
        if direction in ("in", "both"):
            for (r, h) in self.in_edges.get(anchor_qid, ()):
                res.append((h, r, anchor_qid))
        return res

    def existing_among(
        self,
        entity_ids: set[str],
        relation_ids: set[str],
        anchor: str | None = None,
    ) -> list[tuple[str, str, str]]:
        """Existing triples (as label strings) AMONG the retrieved entities/relations,
        BOTH directions. Iterating ordered pairs over out-adjacency catches in-edges too."""
        ents = set(entity_ids)
        if anchor:
            ents.add(anchor)
        rels = set(relation_ids)
        out: list[tuple[str, str, str]] = []
        for e1 in ents:
            for (r, t) in self.out_edges.get(e1, ()):
                if r in rels and t in ents and t != e1:   # skip self-loops (head == tail)
                    out.append((self.label(e1), self.label(r), self.label(t)))
        return out


def _smoke(args: argparse.Namespace) -> None:
    target = {args.anchor} if args.anchor and not args.whole else None
    print(f"loading {args.snapshot} (mode={'whole' if target is None else 'filtered'})...")
    kg = KGLookup.load(args.snapshot, target_anchors=target)
    print(f"loaded {kg.n_triples:,} edges; {len(kg.out_edges):,} heads, "
          f"{len(kg.in_edges):,} tails, {len(kg.labels):,} labels")
    if args.anchor:
        for direction in ("out", "in"):
            edges = kg.edges_of(args.anchor, direction=direction)
            print(f"\n=== anchor {args.anchor} ({kg.label(args.anchor)}) — {len(edges)} {direction} edges ===")
            for (h, r, t) in edges[:12]:
                print(f"   ({kg.label(h)}) --[{kg.label(r)}]--> ({kg.label(t)})   "
                      f"has_triple={kg.has_triple(h, r, t)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smoke-test the KG lookup (out + in edges).")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--anchor", default=None, help="anchor QID to dump (e.g. Q1000001)")
    ap.add_argument("--whole", action="store_true", help="load the whole snapshot (ignore --anchor filter)")
    _smoke(ap.parse_args())
