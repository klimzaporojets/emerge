"""C. Build the `KG context` block and write augmented eval-set instances.

Input-augmentation ONLY: we append a KG-context block to `passage`; the baselines
(KGGen/EDC+) then run unchanged. Two condition families:

  - KG-RAG   : block = retriever top-k entities/relations (+ connectivity triples).
               Gold NOT guaranteed present.
  - oracle@N : block = the gold pieces (per-op, below) GUARANTEED present, mixed with
               N distractors (N in {0,2,4,8,16,32}). The distractors are the
               retriever's MOST-SIMILAR candidates (its top-N to the passage, gold
               excluded) -- NOT random. So oracle@k and KG-RAG@k see the SAME
               retrieved neighbors and differ only in the gold guarantee. Never the
               whole assembled triple for assert-side ops.

Per-op gold pieces the oracle guarantees (existing = non-emerging end(s)):
  EXISTS/DEPRECATE -> the existing triple (both entities + relation)
  ADD              -> both entities + relation
  MINT+ADD         -> the existing entity + relation (emerging end is minted)
  INFER            -> the existing (unmentioned) entity + relation (emerging end minted)

Representation = entities + relation types + KG-connectivity lookup (NO explicit
triple encoding for now). Existing triples shown for EXISTS/DEPRECATE come from the
snapshot connectivity among the listed entities/relations.
"""
from __future__ import annotations

import copy

from kg_context.retriever import Hit
from kg_context.kg_lookup import KGLookup

# TKGU op -> which gold pieces the oracle guarantees
OP_NAMES = {
    "x-triples": "Exists", "e-triples": "Add", "ee-triples": "Mint+Add",
    "ee-kg-triples": "Infer", "d-triples": "Deprecate",
}
_SHOW_EXISTING_TRIPLES = {"x-triples", "d-triples"}  # connectivity section for these

SELECT_MINT_INSTRUCTION = (
    "Candidate existing KG entities are listed with their IDs. When the passage "
    "refers to one, link to it (use its ID); otherwise treat the entity as "
    "new/emerging (mint it, no ID). Use the candidate relation types to connect "
    "entities into KG triples."
)


def gold_pieces(gold_triple: dict) -> dict:
    """Extract the oracle-guaranteed pieces from one gold tkgu triple.

    Uses `triple` = [head_qid, pid, tail_qid], `triple_labels`, and the
    `emerging_head`/`emerging_tail` flags to keep only the EXISTING (non-emerging)
    entities. Returns {entities:{qid:label}, relation:(pid,label), triple:(h,r,t)|None}.
    """
    h, r, t = gold_triple["triple"]
    hl, rl, tl = (gold_triple.get("triple_labels") or [None, None, None])
    ops = gold_triple.get("tkgu_operations") or []
    emerging_head = gold_triple.get("emerging_head", False)
    emerging_tail = gold_triple.get("emerging_tail", False)

    entities: dict[str, str | None] = {}
    if not emerging_head:
        entities[h] = hl
    if not emerging_tail:
        entities[t] = tl
    relation = (r, rl)
    # existing triple only meaningful when both ends exist (EXISTS/DEPRECATE)
    triple = (h, r, t) if (not emerging_head and not emerging_tail) else None
    return {"entities": entities, "relation": relation, "triple": triple, "ops": ops}


def instance_gold_details(instance: dict, target_ops: set[str]) -> dict:
    """From an instance's `tkgu_triples`, pull the target-op gold. Returns the union dicts
    (entities/relations/triples) used to BUILD the oracle block, PLUS a per-triple `gold_list`
    for `kg_context_meta.oracle_gold` (with `gold_triple_idx` into tkgu_triples, both head/tail
    entities carrying role + emerging flag)."""
    entities: dict[str, str | None] = {}
    relations: dict[str, str | None] = {}
    triples: list[tuple[str, str, str]] = []
    ops_present: set[str] = set()
    gold_list: list[dict] = []
    for idx, gt in enumerate(instance.get("tkgu_triples", [])):
        gops = set(gt.get("tkgu_operations") or [])
        if target_ops and not (gops & target_ops):
            continue
        gp = gold_pieces(gt)
        entities.update(gp["entities"])
        pid, plabel = gp["relation"]
        relations[pid] = plabel
        if gp["triple"] is not None:
            triples.append(gp["triple"])
        ops_present |= gops
        h, r, t = gt["triple"]
        hl, rl, tl = (gt.get("triple_labels") or [None, None, None])
        ent_records = [
            {"qid": h, "label": hl, "role": "head", "emerging": bool(gt.get("emerging_head", False))},
            {"qid": t, "label": tl, "role": "tail", "emerging": bool(gt.get("emerging_tail", False))},
        ]
        # ONE oracle_gold entry PER operation -> a single, unambiguous `op` per entry
        ops_for_entries = sorted(gops & target_ops) if target_ops else sorted(gops)
        for op in ops_for_entries:
            gold_list.append({
                "gold_triple_idx": idx,
                "op": op,
                "relation": {"pid": r, "label": rl},
                "entities": [dict(e) for e in ent_records],
                "triple_qids": [h, r, t],
                "triple_labels": [hl, rl, tl],
            })
    return {"entities": entities, "relations": relations, "triples": triples,
            "ops": ops_present, "gold_list": gold_list}


def _fmt(pairs: list[tuple[str, str | None]]) -> list[str]:
    """(id, label) -> 'label (ID)' bullets."""
    out = []
    for cid, lbl in pairs:
        out.append(f"  - {lbl} ({cid})" if lbl else f"  - {cid}")
    return out


def build_kg_block(
    anchor_label: str | None,
    entities: list[tuple[str, str | None]],
    relations: list[tuple[str, str | None]],
    existing_triples: list[tuple[str, str, str]] | None = None,
) -> str:
    lines = ["", "KG context (from the current knowledge graph snapshot).",
             SELECT_MINT_INSTRUCTION, ""]
    lines.append("Candidate relation types:")
    lines += _fmt(relations)
    lines.append("Candidate existing KG entities:")
    lines += _fmt(entities)
    if existing_triples:
        anc = anchor_label or "the anchor entity"
        lines.append(f'Existing KG triples of "{anc}":')
        for (a, r, t) in existing_triples:
            lines.append(f"  - ({a}, {r}, {t})")
    return "\n".join(lines)


def _dedup_keep_order(pairs: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    seen, out = set(), []
    for cid, lbl in pairs:
        if cid not in seen:
            seen.add(cid)
            out.append((cid, lbl))
    return out


def gather_components(
    anchor_qid: str,
    anchor_label: str | None,
    ent_hits: list[Hit],
    rel_hits: list[Hit],
    gold_list: list[dict],
    retriever: str,
    year: str,
    kg: "KGLookup | None" = None,
) -> dict:
    """Gather ALL KG-context components for one passage into a `kg_context` dict. The BLOCK
    is composed later (compose_block) at experiment time; the passage stays RAW.

    Separate lists: `kg_rag_*` (retrieved, WITH cosine scores, ordered by score) vs
    `oracle_*` (gold). `oracle_triples` = the gold EXISTING edges (Exists/Deprecate only;
    both ends existing) -- taken straight from the instance's gold (no snapshot scan needed).
    `gold_retrieved` = did the retriever surface the gold (per pool) -- the key kg_rag signal.

    OPTIONAL `kg` (a loaded KGLookup): if given, also emit `lookup_triples` = the EXISTING
    KG edges among *all* oracle entities via *all* oracle relations (a `KGLookup.existing_among`
    query). Off by default (the core needs no snapshot scan); a future config turns it on to
    experiment with lookup-surfaced existing triples.
    """
    kg_rag_entities = [{"qid": h.id, "label": h.label, "score": round(h.score, 4)} for h in ent_hits]
    kg_rag_relations = [{"pid": h.id, "label": h.label, "score": round(h.score, 4)} for h in rel_hits]

    oracle_entities, seen_e = [], set()
    oracle_relations, seen_r = [], set()
    oracle_triples = []
    for g in gold_list:
        for e in g["entities"]:
            k = (e["qid"], g["op"])
            if e["qid"] and k not in seen_e:
                seen_e.add(k)
                oracle_entities.append({**e, "op": g["op"], "gold_triple_idx": g["gold_triple_idx"]})
        rk = (g["relation"]["pid"], g["op"])
        if rk not in seen_r:
            seen_r.add(rk)
            oracle_relations.append({**g["relation"], "op": g["op"], "gold_triple_idx": g["gold_triple_idx"]})
        if g["op"] in _SHOW_EXISTING_TRIPLES and all(not e["emerging"] for e in g["entities"]):
            oracle_triples.append({"triple_qids": g["triple_qids"], "triple_labels": g["triple_labels"],
                                   "op": g["op"], "gold_triple_idx": g["gold_triple_idx"]})

    ret_ent = {h.id for h in ent_hits}
    ret_rel = {h.id for h in rel_hits}
    gold_ent_ids = {e["qid"] for e in oracle_entities if not e["emerging"]}
    gold_rel_ids = {r["pid"] for r in oracle_relations}

    # OPTIONAL lookup: existing KG edges among all oracle entities via all oracle relations
    lookup_triples = None
    if kg is not None:
        all_ent = {e["qid"] for e in oracle_entities if e["qid"]}
        all_rel = {r["pid"] for r in oracle_relations}
        lookup_triples = kg.existing_among(all_ent, all_rel, anchor=anchor_qid)

    return {
        "retriever": retriever,
        "year": year,
        "anchor_qid": anchor_qid,
        "anchor_label": anchor_label,
        "kg_rag_entities": kg_rag_entities,
        "kg_rag_relations": kg_rag_relations,
        "oracle_entities": oracle_entities,
        "oracle_relations": oracle_relations,
        "oracle_triples": oracle_triples,
        "lookup_triples": lookup_triples,  # None unless KGLookup was provided
        "gold_retrieved": {"entities": bool(gold_ent_ids & ret_ent),
                           "relation": bool(gold_rel_ids & ret_rel)},
    }


def compose_block(
    kgc: dict, condition: str = "oracle_pieces", k: int = 16, seed: int = 0, shuffle: bool = True
) -> str:
    """Build a KG-context block from gathered `kg_context` components for ONE experiment
    config -- this is the experiment-time composer (GENERATE calls it, then appends to the
    passage). condition: 'kg_rag' | 'only_oracle' | 'oracle_pieces' | 'oracle_triples'.
    k = total candidates per pool (entities:k, relations:k). Candidates ordered by retriever
    score; for oracle_pieces, gold + top distractors to k, shuffled (position bias) if `shuffle`.
    """
    import random

    ents_pool = kgc["kg_rag_entities"]
    rels_pool = kgc["kg_rag_relations"]
    gold_ents = [(e["qid"], e["label"]) for e in kgc["oracle_entities"] if not e["emerging"]]
    gold_rels = [(r["pid"], r["label"]) for r in kgc["oracle_relations"]]
    triples = None

    if condition == "kg_rag":
        entities = [(e["qid"], e["label"]) for e in ents_pool[:k]]
        relations = [(r["pid"], r["label"]) for r in rels_pool[:k]]
    elif condition == "only_oracle":
        entities, relations = list(gold_ents), list(gold_rels)
    elif condition == "oracle_pieces":
        gold_eid = {e[0] for e in gold_ents}
        gold_rid = {r[0] for r in gold_rels}
        distr_e = [(e["qid"], e["label"]) for e in ents_pool if e["qid"] not in gold_eid]
        distr_r = [(r["pid"], r["label"]) for r in rels_pool if r["pid"] not in gold_rid]
        entities = _dedup_keep_order(gold_ents + distr_e[: max(0, k - len(gold_ents))])
        relations = _dedup_keep_order(gold_rels + distr_r[: max(0, k - len(gold_rels))])
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(entities)
            rng.shuffle(relations)
    elif condition == "oracle_triples":
        entities, relations = list(gold_ents), list(gold_rels)
        triples = [tuple(t["triple_labels"]) for t in kgc["oracle_triples"]]
    else:
        raise ValueError(f"unknown condition: {condition}")
    return build_kg_block(kgc.get("anchor_label"), entities, relations, triples)


def attach_components(instance: dict, kg_context: dict) -> dict:
    """Deep copy of the instance with the `kg_context` components attached; passage stays RAW.
    GENERATE composes the block (compose_block) at experiment time and appends it before the
    baseline runs. Baselines read only `passage`, so `kg_context` is never seen until composed."""
    aug = copy.deepcopy(instance)
    aug["kg_context"] = kg_context
    return aug
