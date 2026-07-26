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
    hl, rl, tl = (gold_triple["triple_labels"] or [None, None, None])
    ops = gold_triple["tkgu_operations"] or []
    emerging_head = gold_triple["emerging_head"]   # required — missing must NOT silently default to False
    emerging_tail = gold_triple["emerging_tail"]

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
    for idx, gt in enumerate(instance["tkgu_triples"]):
        gops = set(gt["tkgu_operations"] or [])
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
        hl, rl, tl = (gt["triple_labels"] or [None, None, None])
        ent_records = [
            {"qid": h, "label": hl, "role": "head", "emerging": bool(gt["emerging_head"])},
            {"qid": t, "label": tl, "role": "tail", "emerging": bool(gt["emerging_tail"])},
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
    include_entities: bool,          # config-flow: REQUIRED (no defaults)
    include_relations: bool,
    include_instruction: bool,
    existing_triples: list[tuple[str, str, str]] | None = None,   # optional
    instruction: str | None = None,
) -> str:
    lines = ["", "##KG CONTEXT## (from the current knowledge-graph snapshot)"]
    if include_instruction:
        lines.append(instruction or SELECT_MINT_INSTRUCTION)  # per-op template, else the generic default
    lines.append("")
    if include_relations:
        lines.append("Candidate relation types:")
        lines += _fmt(relations)
    if include_entities:
        lines.append("Candidate existing KG entities:")
        lines += _fmt(entities)
    if existing_triples:
        lines.append("Existing KG triples (each element as `label (ID)` — use the IDs in your answer):")
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


INFER_PATTERN_CAP = 6   # Infer Design A: max real (X,P,Q) pattern edges per gold INFER triple (hub guard)


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

    # OPTIONAL lookup (needs a loaded KGLookup): the REAL existing KG edges to show the model.
    lookup_triples = None
    kg_rag_triples = None
    infer_pattern_triples = None
    if kg is not None:
        all_ent = {e["qid"] for e in oracle_entities if e["qid"]}
        all_rel = {r["pid"] for r in oracle_relations}
        lookup_triples = kg.existing_among(all_ent, all_rel, anchor=anchor_qid)

        # kg_rag_triples = real existing edges AMONG the retrieved entities (pairwise, + anchor) via the
        # retrieved relations. The gold/relevant edges connect the passage's entities TO EACH OTHER (not
        # just the anchor), so we scan every retrieved entity's out-edges -> needs each one's adjacency,
        # i.e. a WHOLE-snapshot KGLookup load (kg_lookup.scope='whole').
        kg_rag_triples, seen_kt = [], set()
        ents = ret_ent | {anchor_qid}
        for e1 in ents:
            for (r, t) in kg.out_edges.get(e1, ()):
                if r in ret_rel and t in ents and t != e1 and (e1, r, t) not in seen_kt:  # skip self-loops
                    seen_kt.add((e1, r, t))
                    kg_rag_triples.append({"triple_qids": [e1, r, t],
                                           "triple_labels": [kg.label(e1), kg.label(r), kg.label(t)]})

        # Infer Design A (oracle grounding): real KG edges that share the gold INFER relation P with the
        # NOT-in-text existing entity Q -> other entities that link the SAME way. Direction follows Q's role
        # in the gold triple (Q as tail -> in_edges(Q) via P; Q as head -> out_edges(Q) via P). Skips the
        # minted entity and self-loops; capped per gold triple (hub guard). REAL edges + snapshot labels only.
        infer_pattern_triples, seen_ip = [], set()
        for g in gold_list:
            if g["op"] != "ee-kg-triples":
                continue
            hq, pid, tq = g["triple_qids"]
            minted = {e["qid"] for e in g["entities"] if e["emerging"]}
            for e in g["entities"]:
                q = e["qid"]
                if e["emerging"] or not q:
                    continue
                n = 0
                edges = ((r, x, [x, r, q]) for (r, x) in kg.in_edges.get(q, ())) if q == tq else \
                        ((r, y, [q, r, y]) for (r, y) in kg.out_edges.get(q, ())) if q == hq else iter(())
                for (r, other, trip) in edges:
                    if r == pid and other != q and other not in minted and tuple(trip) not in seen_ip:
                        seen_ip.add(tuple(trip))
                        infer_pattern_triples.append({"triple_qids": trip, "gold_triple_idx": g["gold_triple_idx"],
                            "triple_labels": [kg.label(trip[0]), kg.label(trip[1]), kg.label(trip[2])]})
                        n += 1
                        if n >= INFER_PATTERN_CAP:
                            break

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
        "kg_rag_triples": kg_rag_triples,  # None unless KGLookup provided; real edges among retrieved ents/rels
        "lookup_triples": lookup_triples,  # None unless KGLookup was provided
        "infer_pattern_triples": infer_pattern_triples,  # Infer Design A: real (X,P,Q) edges sharing gold rel+target Q
        "gold_retrieved": {"entities": bool(gold_ent_ids & ret_ent),
                           "relation": bool(gold_rel_ids & ret_rel)},
    }


def _effective_triples(kgc: dict, source: str, op_match, max_kg_rag: int = -1) -> tuple[list, list, list]:
    """qid-annotated (h,r,t) triple tuples for `source` in {'oracle','kg_rag','both'}, plus their
    (id,label) entities/relations. oracle = gold existing edges (op-filtered); kg_rag = edges among
    the retrieved ents/rels (KG-LOOKUP); both = UNION (dedup by (h,r,t) label). Elements annotated
    `label (ID)` so the model can emit canonical QIDs/PIDs (Executable-R).
    `max_kg_rag` caps ONLY the KG-RAG lookup triples (blow-up guard): 0 -> none (oracle-only),
    N>0 -> first N, negative -> unlimited. Oracle ground-truth triples are always kept in full."""
    # drop self-loops (head QID == tail QID, e.g. `United States (Q30) country (P17) United States (Q30)`) --
    # meaningless as candidates. Defensive filter here so current components render clean without a re-gather.
    _noloop = lambda ts: [t for t in ts if t["triple_qids"][0] != t["triple_qids"][2]]
    ot = _noloop([t for t in kgc["oracle_triples"] if op_match(t)])
    kt = _noloop(kgc["kg_rag_triples"] or [])
    # RANK kg_rag candidate triples by the retriever's confidence in their pieces BEFORE capping, so the
    # top-`max_kg_rag` are the highest-confidence edges (not an arbitrary first-N). Score(h,r,t) = the SUM
    # of the retriever score of the head entity + the relation type + the tail entity (from the kg_rag
    # pools; a piece absent from its pool contributes 0). Additive per-passage ranking -> the edges whose
    # ends AND relation the retriever surfaced most confidently come first.
    e_score = {e["qid"]: e.get("score", 0.0) for e in kgc["kg_rag_entities"]}
    r_score = {r["pid"]: r.get("score", 0.0) for r in kgc["kg_rag_relations"]}

    def _triple_score(t: dict) -> float:
        h, r, tl = t["triple_qids"]
        return e_score.get(h, 0.0) + r_score.get(r, 0.0) + e_score.get(tl, 0.0)

    kt = sorted(kt, key=_triple_score, reverse=True)
    if max_kg_rag >= 0:
        kt = kt[:max_kg_rag]
    if source == "oracle":
        src = ot
    elif source == "kg_rag":
        src = kt
    else:  # both -> union, dedup by (h,r,t) QID tuple (robust: same edge appears ONCE even if the
           # oracle source and the KG-LOOKUP source render its labels slightly differently).
        src, seen = [], set()
        for t in ot + kt:
            key = tuple(t["triple_qids"])
            if key not in seen:
                seen.add(key)
                src.append(t)
    tri, ents, rels = [], [], []
    for td in src:
        q, l = td["triple_qids"], td["triple_labels"]
        tri.append((f"{l[0]} ({q[0]})", f"{l[1]} ({q[1]})", f"{l[2]} ({q[2]})"))
        ents += [(q[0], l[0]), (q[2], l[2])]
        rels.append((q[1], l[1]))
    return tri, ents, rels


_TRIPLE_COND = {"oracle_triples": "oracle", "kg_rag_triples": "kg_rag", "both_triples": "both"}


def select_candidates(
    kgc: dict, condition: str, k: int, seed: int, order: str, max_triples: int,
    op: str | None = None, include_triples: str | None = None,
) -> tuple[list, list, list]:
    """Pick the (entities, relations, triples) to show for ONE condition -- the single source of truth
    for both the legacy block builder and the on-disk template renderer. entities [(qid,label)],
    relations [(pid,label)], triples [(head'label (ID)', rel'label (ID)', tail'label (ID)')].
    condition: text_only | kg_rag | only_oracle | oracle_pieces | oracle_triples | kg_rag_triples | both_triples.
    `order='shuffle'` re-shuffles oracle_pieces (position-bias check). `op` -> that operation's gold pieces
    only (kg_rag pools are op-agnostic). max_triples: -1 unlimited / 0 none / N first-N (kg_rag lookup only)."""
    import random
    if condition == "text_only":
        return [], [], []

    def _op_match(x) -> bool:
        return op is None or x["op"] == op

    ents_pool = kgc["kg_rag_entities"]
    rels_pool = kgc["kg_rag_relations"]
    gold_ents = [(e["qid"], e["label"]) for e in kgc["oracle_entities"] if not e["emerging"] and _op_match(e)]
    gold_rels = [(r["pid"], r["label"]) for r in kgc["oracle_relations"] if _op_match(r)]
    triples: list = []

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
    elif condition in _TRIPLE_COND:
        # effective REAL KG triples (Exists/Deprecate). source = oracle | kg_rag | both(union).
        source = _TRIPLE_COND[condition]
        triples, t_ents, t_rels = _effective_triples(kgc, source, _op_match, max_kg_rag=max_triples)
        if source == "kg_rag":
            entities, relations = _dedup_keep_order(t_ents), _dedup_keep_order(t_rels)  # retrieved only, no gold leak
        else:  # oracle / both include the gold pieces
            entities = _dedup_keep_order(list(gold_ents) + t_ents)
            relations = _dedup_keep_order(list(gold_rels) + t_rels)
    else:
        raise ValueError(f"unknown condition: {condition}")

    # `include_triples`: ALSO show existing triples on top of a PIECES condition (Add/Mint/Infer, and
    # unified) -> the prompt has candidates AND real KG edges as CONTEXT. Op-AGNOSTIC (the oracle edges
    # are the instance's Exists/Deprecate triples, not this op's gold). `max_triples` caps the count.
    if include_triples and condition not in _TRIPLE_COND:
        extra_tri, extra_e, extra_r = _effective_triples(kgc, include_triples, lambda _x: True,
                                                         max_kg_rag=max_triples)
        triples = (triples or []) + extra_tri
        entities = _dedup_keep_order(entities + extra_e)
        relations = _dedup_keep_order(relations + extra_r)
    # SHUFFLE the 'both' (oracle+kg_rag) conditions so the oracle (correct) pieces are NOT positionally
    # distinguishable from the kg_rag distractors -> the LLM can't guess the answer by order. Shuffles
    # entities, relations AND triples. Deterministic (seeded). Also honored when order=='shuffle'
    # (explicit position-bias probe). oracle-only / kg_rag-only keep their meaningful order (gold / score).
    if order == "shuffle" or condition in ("both_triples", "oracle_pieces"):
        rng = random.Random(seed)
        rng.shuffle(entities)
        rng.shuffle(relations)
        rng.shuffle(triples)
    return entities, relations, triples


def compose_block(
    kgc: dict, condition: str, k: int, seed: int, order: str,
    include_entities: bool, include_relations: bool, include_instruction: bool, max_triples: int,
    op: str | None = None, instruction: str | None = None, include_triples: str | None = None,
) -> str:
    """[kept for the AUGMENT inspect-preview] assemble the KG-context block string from the selected
    candidates. 'text_only' returns ''. Template rendering uses `template_fields` + `fill_template`."""
    if condition == "text_only":
        return ""
    entities, relations, triples = select_candidates(kgc, condition, k, seed, order, max_triples,
                                                      op=op, include_triples=include_triples)
    return build_kg_block(
        kgc["anchor_label"], entities, relations,
        include_entities=include_entities, include_relations=include_relations,
        include_instruction=include_instruction, existing_triples=triples, instruction=instruction)


# ---- on-disk prompt templates (src/kg_context/prompt_templates_kg_aware/): load + fill placeholders ----
def _section(header: str, pairs: list) -> str:
    return (header + "\n" + "\n".join(_fmt(pairs))) if pairs else ""


def _triples_section(triples: list) -> str:
    if not triples:
        return ""
    head = "Existing KG triples (each element as `label (ID)` — use the IDs in your answer):"
    return head + "\n" + "\n".join(f"  - ({a}, {r}, {t})" for (a, r, t) in triples)


def _default_fields(
    kgc: dict, condition: str, k: int, seed: int, order: str,
    include_entities: bool, include_relations: bool, max_triples: int,
    op: str | None = None, include_triples: str | None = None,
) -> dict:
    """Generic placeholder strings (each '' when absent -> empty sections vanish in `fill_template`).
    Keys: relation_types, entities, existing_triples, kg_context. Used by every op that has no special
    builder in OP_FIELD_BUILDERS."""
    entities, relations, triples = select_candidates(kgc, condition, k, seed, order, max_triples,
                                                      op=op, include_triples=include_triples)
    rt = _section("Candidate relation types:", relations) if include_relations else ""
    en = _section("Candidate existing KG entities:", entities) if include_entities else ""
    tr = _triples_section(triples)
    # Negative / empty-KG case (e.g. an oracle prompt for an op this instance does NOT have): nothing was
    # retrieved. Emit an EXPLICIT notice so the ##KG CONTEXT## section survives `_tidy` instead of vanishing
    # and leaving the task text ("choose from the triples listed below") pointing at nothing. The correct
    # answer here is [] -- this is the hallucination/negative test. EXCLUDE `text_only`: that condition shows
    # NO KG context BY DESIGN (the no-KG baseline), so its section must stay absent, not carry a notice.
    if condition != "text_only" and not (rt or en or tr):
        tr = "NO CANDIDATE TRIPLES OR ENTITIES WERE FOUND IN THE KG FOR THIS PASSAGE."
    kg = "\n".join(x for x in ["##KG CONTEXT## (from the current knowledge-graph snapshot)", rt, en, tr] if x)
    return {"relation_types": rt, "entities": en, "existing_triples": tr, "kg_context": kg}


def _infer_fields(
    kgc: dict, condition: str, k: int, seed: int, order: str,
    include_entities: bool, include_relations: bool, max_triples: int,
    op: str | None = None, include_triples: str | None = None,
) -> dict:
    """Infer field builder = generic fields PLUS `{pattern_examples}` (Design A). Pattern edges are the
    instance's real (X,P,Q) edges sharing the gold INFER relation+target Q (gathered as
    `infer_pattern_triples`). Shown ONLY for the ORACLE conditions (only_oracle/oracle_pieces) -- the
    gold-grounded upper-bound signal; empty (section drops) for kg_rag/text_only."""
    f = _default_fields(kgc, condition, k, seed, order, include_entities, include_relations, max_triples,
                        op=op, include_triples=include_triples)
    pats = kgc["infer_pattern_triples"] if "infer_pattern_triples" in kgc else None
    if condition in ("only_oracle", "oracle_pieces") and pats:
        head = ("For this particular passage, here are some closely-related examples that can help you (real "
                "existing KG edges that use a candidate relation with a candidate KG entity — the NEW in-text "
                "entity most likely links the SAME way as these):")
        rows = [f"  - ({tl[0]} ({q[0]}), {tl[1]} ({q[1]}), {tl[2]} ({q[2]}))"
                for t in pats for q, tl in [(t["triple_qids"], t["triple_labels"])]]
        f["pattern_examples"] = head + "\n" + "\n".join(rows)
    else:
        f["pattern_examples"] = ""
    return f


# op_label -> field builder. Ops NOT listed here fall back to `_default_fields`. Register an op's OWN
# builder when it needs DIFFERENT fields / logic than the generic set -- e.g. Infer adds relation-pattern
# demos ({pattern_examples}), Mint+Add may want its own entity handling, etc. A per-op template may then
# reference op-specific placeholders; `fill_template` HARD-FAILS if the op's builder doesn't provide a
# placeholder the template uses (so a missing per-op field is caught, not silent).
OP_FIELD_BUILDERS: dict = {"infer": _infer_fields}


def template_fields(
    op_label: str, kgc: dict, condition: str, k: int, seed: int, order: str,
    include_entities: bool, include_relations: bool, max_triples: int,
    op: str | None = None, include_triples: str | None = None,
) -> dict:
    """Dispatch to op_label's field builder (default: `_default_fields`) -> {placeholder: rendered str}."""
    builder = OP_FIELD_BUILDERS[op_label] if op_label in OP_FIELD_BUILDERS else _default_fields
    return builder(kgc, condition, k, seed, order, include_entities, include_relations, max_triples,
                   op=op, include_triples=include_triples)


def _tidy(text: str) -> str:
    """Drop any '##SECTION##' header whose body (up to the next '##' header / EOF) is empty, then
    collapse blank-line runs. Lets a template list all sections; the absent ones disappear cleanly."""
    import re
    is_hdr = lambda s: re.match(r"^##.+?##", s.strip()) is not None
    lines = text.split("\n")
    keep, i = [], 0
    while i < len(lines):
        if is_hdr(lines[i]):
            j, body = i + 1, False
            while j < len(lines) and not is_hdr(lines[j]):
                if lines[j].strip():
                    body = True
                j += 1
            if not body:
                i = j
                continue
        keep.append(lines[i])
        i += 1
    return re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip() + "\n"


def fill_template(template: str, passage: str, fields: dict, output_format: str,
                  few_shot_examples: str = "") -> str:
    """Substitute {passage}, {output_format}, {few_shot_examples}, and every key of `fields` (the op's
    placeholders) into the template, then tidy empty sections. HARD-FAILS if the template still references a
    {placeholder} the op's field builder did not provide -- a template asking for a field it doesn't compute
    is a bug. `few_shot_examples` empty -> the ##EXAMPLES## section auto-drops (zero-shot)."""
    import re
    subs = {"passage": passage, "output_format": output_format,
            "few_shot_examples": few_shot_examples, **fields}
    out = template
    for key, val in subs.items():
        out = out.replace("{" + key + "}", val)
    leftover = sorted(set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", out)))
    if leftover:
        raise KeyError(f"template has unfilled placeholder(s) {leftover} not provided by this op's field "
                       f"builder (available: {sorted(subs)}). Register an OP_FIELD_BUILDERS entry or fix the template.")
    return _tidy(out)


def attach_components(instance: dict, kg_context: dict) -> dict:
    """Deep copy of the instance with the `kg_context` components attached; passage stays RAW.
    GENERATE composes the block (compose_block) at experiment time and appends it before the
    baseline runs. Baselines read only `passage`, so `kg_context` is never seen until composed."""
    aug = copy.deepcopy(instance)
    aug["kg_context"] = kg_context
    return aug
