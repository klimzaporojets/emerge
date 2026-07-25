#!/usr/bin/env python3
"""Gold-retrieval coverage report (recall@k) over a GATHER output dir.

For each component file (raw passage + kg_context), we already stored the retrieved candidates
`kg_rag_entities` / `kg_rag_relations` ORDERED by retriever score. The gold set is the same one
augment.py uses for `gold_retrieved`:
  gold entities  = oracle_entities with emerging == False  (the existing, retrievable ends)
  gold relations = all oracle_relations
recall@k (per pool) = is any gold id present in the TOP-k of the ordered candidate list?
We report recall@k for k in 1,2,4,8,16,32,64 (capped at retrieve_top_k), per year and overall.

Denominator = instances that HAVE a retrievable gold in that pool (entity pool skips instances
whose only gold ends are emerging -> nothing to retrieve). Relation pool denom = all instances.

    PYTHONPATH=src python scripts/kg_context/coverage_report.py \
        output/kg_context/20260724_local_fewshot_relik
"""
from __future__ import annotations

import glob
import json
import os
import sys

KS = [1, 2, 4, 8, 16, 32, 64]
OP_LABEL = {"x-triples": "exists", "e-triples": "add", "ee-triples": "mint_add",
            "ee-kg-triples": "infer", "d-triples": "deprecate"}


def _year(fp: str) -> str:
    for part in fp.split(os.sep):
        if part.startswith("snapshot_"):
            return part[len("snapshot_"):][:4]
    return "????"


def _first_hit_rank(ordered_ids: list[str], gold: set[str]) -> int | None:
    """0-based rank of the FIRST gold id in the ordered candidate list, or None if absent."""
    for i, cid in enumerate(ordered_ids):
        if cid in gold:
            return i
    return None


def load_rows(out_dir: str) -> list[dict]:
    rows = []
    for fp in sorted(glob.glob(os.path.join(out_dir, "snapshot_*", "delta_*.jsonl"))):
        yr = _year(fp)
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                inst = json.loads(line)
                kgc = inst["kg_context"]
                gold_ent = {e["qid"] for e in kgc["oracle_entities"]
                            if e["qid"] and not e["emerging"]}
                gold_rel = {r["pid"] for r in kgc["oracle_relations"] if r["pid"]}
                ent_ids = [e["qid"] for e in kgc["kg_rag_entities"]]
                rel_ids = [r["pid"] for r in kgc["kg_rag_relations"]]
                # per-op rows too (gold restricted to that op) -> one extra row per op present
                by_op = []
                for op in OP_LABEL:
                    g_e = {e["qid"] for e in kgc["oracle_entities"]
                           if e["op"] == op and e["qid"] and not e["emerging"]}
                    g_r = {r["pid"] for r in kgc["oracle_relations"]
                           if r["op"] == op and r["pid"]}
                    if not g_e and not g_r:
                        continue
                    by_op.append({
                        "yr": yr, "op": op,
                        "ent_rank": _first_hit_rank(ent_ids, g_e) if g_e else None,
                        "rel_rank": _first_hit_rank(rel_ids, g_r) if g_r else None,
                        "has_ent_gold": bool(g_e), "has_rel_gold": bool(g_r),
                        "depth": max(len(ent_ids), len(rel_ids)),
                    })
                rows.append({
                    "yr": yr,
                    "gold_ent": gold_ent, "gold_rel": gold_rel,
                    "ent_rank": _first_hit_rank(ent_ids, gold_ent) if gold_ent else None,
                    "rel_rank": _first_hit_rank(rel_ids, gold_rel) if gold_rel else None,
                    "has_ent_gold": bool(gold_ent), "has_rel_gold": bool(gold_rel),
                    "depth": max(len(ent_ids), len(rel_ids)),
                    "by_op": by_op,
                })
    return rows


def report_by_op(rows: list[dict], ks: list[int]) -> None:
    op_rows = [r for row in rows for r in row["by_op"]]
    if not op_rows:
        return
    for pool, rank_key, has_key in [("ENTITY", "ent_rank", "has_ent_gold"),
                                    ("RELATION", "rel_rank", "has_rel_gold")]:
        print(f"=== {pool} recall@k BY OPERATION ===")
        print("  " + f"{'op':10} " + " ".join(f"@{k:<5}" for k in ks) + "  n")
        for op, label in OP_LABEL.items():
            sel = [r for r in op_rows if r["op"] == op and r[has_key]]
            if not sel:
                continue
            cells = [f"{(100*sum(1 for r in sel if r[rank_key] is not None and r[rank_key] < k)//len(sel)):>3}% "
                     for k in ks]
            print(f"  {label:10} " + " ".join(cells) + f"  {len(sel)}")
        print()


def recall_at(rows: list[dict], rank_key: str, has_key: str, k: int) -> tuple[int, int]:
    """(#hit@k, #denom) -- denom = instances with a retrievable gold in this pool."""
    denom = [r for r in rows if r[has_key]]
    hit = sum(1 for r in denom if r[rank_key] is not None and r[rank_key] < k)
    return hit, len(denom)


def report(out_dir: str) -> None:
    rows = load_rows(out_dir)
    if not rows:
        print(f"no component files under {out_dir}")
        return
    depth = max((r["depth"] for r in rows), default=0)
    ks = [k for k in KS if k <= depth] or [depth]
    years = sorted({r["yr"] for r in rows})

    print(f"coverage over {len(rows)} instances (retrieved depth={depth}) -- {out_dir}\n")
    for pool, rank_key, has_key in [("ENTITY", "ent_rank", "has_ent_gold"),
                                    ("RELATION", "rel_rank", "has_rel_gold")]:
        print(f"=== {pool} recall@k ===")
        header = "  " + f"{'scope':6} " + " ".join(f"@{k:<5}" for k in ks) + "  denom"
        print(header)
        for scope in years + ["ALL"]:
            sel = rows if scope == "ALL" else [r for r in rows if r["yr"] == scope]
            cells = []
            denom = 0
            for k in ks:
                hit, denom = recall_at(sel, rank_key, has_key, k)
                cells.append(f"{(100*hit//denom) if denom else 0:>3}% ")
            print(f"  {scope:6} " + " ".join(cells) + f"  {denom}")
        print()

    # per-operation breakdown (retrieval difficulty tracks op nature: emerging-end ops are hard)
    report_by_op(rows, ks)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "output/kg_context/20260724_local_fewshot_relik"
    report(out)
