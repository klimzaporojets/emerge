#!/usr/bin/env python3
"""Build augmented eval-set copies (KG-RAG + oracle distractor sweep) from a config.

CORRECTED DESIGN: candidates come from an INDEX-WIDE dense retriever (all-mpnet index
from index_builder.py, or ReLiK later) -- no traversal. The snapshot is used only for
a membership check (which retrieved entity/relation pairs actually connect to the
anchor, to show existing triples for EXISTS/DEPRECATE).

Chains INDEX(membership) + RETRIEVE(dense) + AUGMENT. Writes:
  <output_dir>/kg_rag/snapshot_<year>/<delta>.jsonl        (if kg_rag.enabled)
  <output_dir>/oracle_k<N>/snapshot_<year>/<delta>.jsonl   (per distractor level N)
  <output_dir>/inspect.txt                                 (human-readable blocks)

    PYTHONPATH=src python scripts/kg_context/build_augmented_evalset.py \
        --config config/kg_context/20260724_local_fewshot/config.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

from tqdm import tqdm

from kg_context.kg_lookup import KGLookup
from kg_context import augment as A

SHOW_TRIPLE_OPS = {"x-triples", "d-triples"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def instance_year(inst: dict, fallback_from_path: str) -> str:
    dates = inst.get("delta_dates") or []
    if dates:
        return str(dates[0])[:4]
    for part in fallback_from_path.split(os.sep):
        if part.startswith("snapshot_"):
            return part[len("snapshot_"):][:4]
    return "unknown"


def select_sample(cfg: dict) -> list[tuple[str, str, dict]]:
    """Return [(year, source_relpath, instance)] for the configured sample."""
    s = cfg["sample"]
    want_years = set(s.get("years") or [])
    want_ops = set(s.get("operations") or [])
    want_ids = set(s.get("hash_ids") or [])
    eval_dir = cfg["eval_set_dir"]

    pool: list[tuple[str, str, dict]] = []
    for fp in sorted(glob.glob(os.path.join(eval_dir, "snapshot_*", "delta_*.jsonl"))):
        relpath = os.path.relpath(fp, eval_dir)
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                inst = json.loads(line)
                yr = instance_year(inst, fp)
                if want_years and yr not in want_years:
                    continue
                ops = {o for gt in inst.get("tkgu_triples", [])
                       for o in (gt.get("tkgu_operations") or [])}
                if want_ops and not (ops & want_ops):
                    continue
                if want_ids:
                    if inst.get("hash_id") in want_ids:
                        pool.append((yr, relpath, inst))
                    continue
                pool.append((yr, relpath, inst))

    if want_ids:
        return pool
    n = s.get("n_instances", len(pool))
    rng = random.Random(s.get("seed", 0))
    rng.shuffle(pool)
    return pool[:n]


def write_instance(out_root: str, tag: str, relpath: str, inst: dict) -> None:
    dest = os.path.join(out_root, tag, relpath)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "a") as f:  # append: sample instances may share a delta file
        f.write(json.dumps(inst) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    target_ops = set(cfg["sample"].get("operations") or [])
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    depth = cfg.get("retrieve_top_k", 32)   # candidates stored per pool; experiment k's are prefixes

    # KG-LOOKUP is OPTIONAL (config `kg_lookup.enabled`) -- OFF by default (the core needs no
    # snapshot scan). ON => also gather the `lookup_triples` component (existing edges among ALL
    # oracle entities/relations).
    kgl_cfg = cfg.get("kg_lookup", {})
    use_lookup = bool(kgl_cfg.get("enabled"))
    scope = kgl_cfg.get("scope", "anchors")   # "whole" | "anchors" (fast)

    sample = select_sample(cfg)
    if args.limit:
        sample = sample[: args.limit]
    print(f"selected {len(sample)} instances; retrieve_top_k={depth}; kg_lookup={use_lookup}")

    # WHICH years to load = config sample.years (already applied when selecting `sample`).
    # Process ONE year at a time: load that year's KG-LOOKUP + retriever, do its instances, then
    # RELEASE before the next year. Peak RAM = a single year's index -- critical for ReLiK
    # (~15 GB/yr), so N years never means N x 15 GB in memory.
    import gc
    from collections import OrderedDict

    by_year: "OrderedDict[str, list]" = OrderedDict()
    for item in sample:
        by_year.setdefault(item[0], []).append(item)

    r = cfg["retriever"]
    which = r.get("which")

    shared_enc = None
    if which == "mpnet_index":
        from kg_context.retriever import MpnetIndexRetriever
        shared_enc = MpnetIndexRetriever.shared_model(r["encoder"], r.get("device", "cpu"))

    def build_retriever(year: str):
        if which == "mpnet_index":
            from kg_context.retriever import MpnetCandidateRetriever
            base = r["index_dir_template"].format(year=year)
            print(f"loading all-mpnet index {base} ...")
            return MpnetCandidateRetriever(
                os.path.join(base, "entities"), os.path.join(base, "relations"), model=shared_enc)
        if which == "relik":
            from kg_context.retriever import RelikCandidateRetriever
            date = f"{year}-01-01"  # standard yearly snapshots; sub-snapshots (2024-04/07) TBD
            print(f"loading ReLiK index for {date} ...")
            return RelikCandidateRetriever(
                config_name=r["config_name"],
                entity_index_path=os.path.join(r["entity_index_base_path"], date),
                relation_index_path=os.path.join(r["relation_index_base_path"], date),
                precision=r.get("precision", "16"), device=r.get("device", "cpu"),
                cache_dir=r.get("cache_dir"))
        raise NotImplementedError(f"retriever '{which}' unknown (use 'mpnet_index' or 'relik')")

    inspect: list[str] = []
    n_written = 0
    for yr, items in by_year.items():
        anchors = {inst["anchor_page_qid"] for (_, _, inst) in items}
        kg = None
        if use_lookup:
            path = os.path.join(cfg["snapshot_dir"], cfg["snapshot_file_template"].format(year=yr))
            tgt = None if scope == "whole" else anchors
            print(f"KG-LOOKUP: loading {path} "
                  f"({'WHOLE ~1-2 min' if tgt is None else f'{len(anchors)} anchors (fast)'})...")
            kg = KGLookup.load(path, target_anchors=tgt)
            print(f"  -> KG-LOOKUP: {kg.n_triples:,} edges over {len(kg.out_edges):,} anchors")
        retriever = build_retriever(yr)

        for (_, relpath, inst) in tqdm(items, desc=f"gather {yr}", unit="inst"):
            anchor = inst["anchor_page_qid"]
            anchor_label = inst.get("anchor_title") or anchor
            passage = inst["passage"]
            ent_hits = retriever.retrieve_entities(passage, depth)
            rel_hits = retriever.retrieve_relations(passage, depth)
            gold = A.instance_gold_details(inst, target_ops)
            kgc = A.gather_components(anchor, anchor_label, ent_hits, rel_hits,
                                     gold["gold_list"], which, yr, kg=kg)
            # ONE component file per passage: raw passage + kg_context components (no block yet).
            write_instance(out_dir, "", relpath, A.attach_components(inst, kgc))
            n_written += 1

            # inspect previews: compose a few example blocks so we can eyeball what GENERATE would show
            seed = (cfg["sample"].get("seed", 0) + n_written) & 0xFFFFFFFF
            inspect.append(f"\n{'='*90}\n[{n_written-1}] {inst['hash_id']} anchor={anchor} ({anchor_label}) "
                           f"ops={sorted(gold['ops'])} gold_retrieved={kgc['gold_retrieved']}\nPASSAGE: {passage[:280]}")
            previews = [("kg_rag", 8), ("only_oracle", None), ("oracle_pieces", 8)]
            if kgc["oracle_triples"]:
                previews.append(("oracle_triples", None))
            for cond, kk in previews:
                block = A.compose_block(kgc, condition=cond, k=(kk or 8), seed=seed)
                inspect.append(f"\n--- compose {cond}{('@'+str(kk)) if kk else ''} ---\n{block}")

        # release this year's heavy objects (esp. ReLiK's ~15 GB index) before the next year
        del retriever, kg
        gc.collect()

    with open(os.path.join(out_dir, "inspect.txt"), "w") as f:
        f.write("\n".join(inspect))
    print(f"done. {n_written} component files + inspect.txt under {out_dir}")


if __name__ == "__main__":
    main()
