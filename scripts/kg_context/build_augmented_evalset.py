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
    want_years = set(s["years"])          # required config keys -> HARD FAIL if absent (no silent defaults)
    want_ops = set(s["operations"])
    want_ids = set(s["hash_ids"])
    eval_dir = cfg["eval_set_dir"]

    pool: list[tuple[str, str, dict]] = []
    files = sorted(glob.glob(os.path.join(eval_dir, "snapshot_*", "delta_*.jsonl")))

    def _file_year(fp: str) -> str | None:
        for part in fp.split(os.sep):
            if part.startswith("snapshot_"):
                return part[len("snapshot_"):][:4]
        return None

    # Filter files by year from the PATH before parsing: eval instances are large (each carries
    # 13 models' predictions), so skipping non-target years up front avoids parsing 5/7 of the set.
    if want_years:
        files = [fp for fp in files if _file_year(fp) in want_years]

    for fp in tqdm(files, desc="scan eval-set", unit="file"):
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
                ops = {o for gt in inst["tkgu_triples"] for o in (gt["tkgu_operations"] or [])}
                if want_ops and not (ops & want_ops):
                    continue
                if want_ids:
                    if inst["hash_id"] in want_ids:
                        pool.append((yr, relpath, inst))
                    continue
                pool.append((yr, relpath, inst))

    if want_ids:
        return pool
    n = s["n_instances"]                  # required (no silent 'all' default)
    rng = random.Random(s["seed"])        # required
    rng.shuffle(pool)
    return pool[:n]


def write_instance(out_root: str, tag: str, relpath: str, inst: dict) -> None:
    dest = os.path.join(out_root, tag, relpath)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "a") as f:  # append: sample instances may share a delta file
        f.write(json.dumps(inst) + "\n")


def _merge_inspect(out_dir: str) -> None:
    """Concatenate per-year inspect_<year>.txt fragments -> inspect.txt (year order)."""
    parts = sorted(glob.glob(os.path.join(out_dir, "inspect_*.txt")))
    with open(os.path.join(out_dir, "inspect.txt"), "w") as out:
        for p in parts:
            with open(p) as f:
                out.write(f.read())


def gather_worker(cfg: dict, limit: int | None) -> int:
    """Retrieve candidates + gather components for the year(s) in `cfg`. In subprocess-per-year mode
    `cfg` holds exactly ONE year, so this whole PROCESS exits right after -> the OS reclaims the
    ~18 GB index fully. (Python's `del`+`gc.collect()` free the objects logically but do NOT return
    a 6M-entry parsed index to the OS -- only a process exit does, which is why years are forked.)
    Writes ONE component file per passage + a per-year `inspect_<year>.txt`. Returns #files written."""
    target_ops = set(cfg["sample"]["operations"])
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    depth = cfg["retrieve_top_k"]   # REQUIRED (no silent 32 default) — candidates stored per pool

    # KG-LOOKUP config is REQUIRED to be present (its `enabled` flag toggles the snapshot scan +
    # `lookup_triples`/`kg_rag_triples` gathering). Hard-fail rather than silently defaulting.
    kgl_cfg = cfg["kg_lookup"]
    use_lookup = bool(kgl_cfg["enabled"])
    scope = kgl_cfg["scope"]        # REQUIRED "whole" | "anchors" (no silent default)

    sample = select_sample(cfg)
    if limit:
        sample = sample[:limit]
    print(f"selected {len(sample)} instances; retrieve_top_k={depth}; kg_lookup={use_lookup}", flush=True)

    import gc
    from collections import OrderedDict

    by_year: "OrderedDict[str, list]" = OrderedDict()
    for item in sample:
        by_year.setdefault(item[0], []).append(item)

    r = cfg["retriever"]
    which = r["which"]

    shared_enc = None
    if which == "mpnet_index":
        from kg_context.retriever import MpnetIndexRetriever
        print(f"loading encoder {r['encoder']} ...", flush=True)
        shared_enc = MpnetIndexRetriever.shared_model(r["encoder"], r["device"])

    def build_retriever(year: str):
        if which == "mpnet_index":
            from kg_context.retriever import MpnetCandidateRetriever
            base = r["index_dir_template"].format(year=year)
            print(f"loading all-mpnet index {base} ...", flush=True)
            return MpnetCandidateRetriever(
                os.path.join(base, "entities"), os.path.join(base, "relations"), model=shared_enc)
        if which == "relik":
            from kg_context.retriever import RelikCandidateRetriever
            date = f"{year}-01-01"  # standard yearly snapshots; sub-snapshots (2024-04/07) TBD
            print(f"loading ReLiK index for {date} ...", flush=True)
            return RelikCandidateRetriever(
                config_name=r["config_name"],
                entity_index_path=os.path.join(r["entity_index_base_path"], date),
                relation_index_path=os.path.join(r["relation_index_base_path"], date),
                precision=r["precision"], device=r["device"],
                cache_dir=r["cache_dir"],
                entity_config_name=(r["entity_config_name"] if "entity_config_name" in r else None))  # optional: None=native
        raise NotImplementedError(f"retriever '{which}' unknown (use 'mpnet_index' or 'relik')")

    n_written = 0
    for yr, items in by_year.items():
        anchors = {inst["anchor_page_qid"] for (_, _, inst) in items}
        kg = None
        if use_lookup:
            path = os.path.join(cfg["snapshot_dir"], cfg["snapshot_file_template"].format(year=yr))
            tgt = None if scope == "whole" else anchors
            print(f"KG-LOOKUP: loading {path} "
                  f"({'WHOLE ~1-2 min' if tgt is None else f'{len(anchors)} anchors (fast)'})...", flush=True)
            kg = KGLookup.load(path, target_anchors=tgt)
            print(f"  -> KG-LOOKUP: {kg.n_triples:,} edges over {len(kg.out_edges):,} anchors", flush=True)
        retriever = build_retriever(yr)

        if which == "relik":
            print(f"gathering {len(items)} passages for {yr} -- retriever-only (reader skipped); "
                  f"the FIRST passage warms up the encoders, later passages are fast.", flush=True)
        inspect: list[str] = []   # per-year fragment (flushed to inspect_<year>.txt below)
        for i, (_, relpath, inst) in enumerate(tqdm(items, desc=f"gather {yr}", unit="inst")):
            anchor = inst["anchor_page_qid"]
            anchor_label = inst["anchor_title"] or anchor
            passage = inst["passage"]
            if which == "relik" and i == 0:
                print(f"  [{yr}] first passage -> encoder forward (span + triplet retrievers) ...", flush=True)
            ent_hits = retriever.retrieve_entities(passage, depth)
            rel_hits = retriever.retrieve_relations(passage, depth)
            if which == "relik" and i == 0:
                print(f"  [{yr}] first passage done: {len(ent_hits)} entity + {len(rel_hits)} relation candidates.",
                      flush=True)
            gold = A.instance_gold_details(inst, target_ops)
            kgc = A.gather_components(anchor, anchor_label, ent_hits, rel_hits,
                                     gold["gold_list"], which, yr, kg=kg)
            # ONE component file per passage: raw passage + kg_context components (no block yet).
            write_instance(out_dir, "", relpath, A.attach_components(inst, kgc))
            n_written += 1

            # inspect previews: compose a few example blocks so we can eyeball what GENERATE would show
            seed = (cfg["sample"]["seed"] + len(inspect) + 1) & 0xFFFFFFFF
            inspect.append(f"\n{'='*90}\n[{yr}#{i}] {inst['hash_id']} anchor={anchor} ({anchor_label}) "
                           f"ops={sorted(gold['ops'])} gold_retrieved={kgc['gold_retrieved']}\nPASSAGE: {passage[:280]}")
            previews = [("kg_rag", 8), ("only_oracle", None), ("oracle_pieces", 8)]
            if kgc["oracle_triples"]:
                previews.append(("oracle_triples", None))
            for cond, kk in previews:
                # explicit preview args (no reliance on compose_block defaults); real prompts come from COMPOSE
                block = A.compose_block(kgc, condition=cond, k=(kk or 8), seed=seed, order="score",
                                        include_entities=True, include_relations=True,
                                        include_instruction=True, op=None, max_triples=32)
                inspect.append(f"\n--- compose {cond}{('@'+str(kk)) if kk else ''} ---\n{block}")

        # release this year's heavy objects (frees them in Python; the OS reclaim happens on process exit)
        del retriever, kg
        gc.collect()

        with open(os.path.join(out_dir, f"inspect_{yr}.txt"), "w") as f:
            f.write("\n".join(inspect))

        # BIG, unmistakable per-year DONE banner -> clear in the log when a year is finished.
        bar = "#" * 78
        print(
            "\n" + bar
            + f"\n#####   YEAR {yr} DONE  —  {len(items)} component files written   #####"
            + f"\n#####   -> {out_dir}/snapshot_{yr}-01-01/   (inspect_{yr}.txt written)"
            + "\n" + bar + "\n",
            flush=True,
        )
    return n_written


def main() -> None:
    # Line-buffer stdout so every print() shows up IMMEDIATELY even through `| tee` (otherwise
    # stdout is block-buffered -> logs surface minutes late, esp. behind the heavy ReLiK load).
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--single", action="store_true",
                    help="internal: run this (single-year) config in-process; used by the per-year driver")
    ap.add_argument("--no-subprocess", action="store_true",
                    help="force in-process multi-year (NOT recommended: RAM accumulates across years)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["output_dir"]
    years = list(dict.fromkeys(cfg["sample"]["years"]))

    # DRIVER: multi-year config -> fork ONE SUBPROCESS PER YEAR so each year's ~18 GB index is fully
    # reclaimed by the OS on child exit (peak RAM = a single year, no accumulation). We sample ONCE
    # here (deterministic, no index loaded), split the selection by year, and hand each child a temp
    # config pinned to its year + hash_ids -> the global n-instance sample is preserved exactly.
    if len(years) > 1 and not args.single and not args.no_subprocess:
        import subprocess
        import tempfile
        from collections import OrderedDict

        # START CLEAN. Delta files are opened in APPEND mode ('a' in write_instance: sampled instances
        # may share a delta file WITHIN one run). So re-running over a previous run's output would DOUBLE
        # every record (old + new mixed). Wipe the whole output dir ONCE here in the PARENT — never in a
        # `--single` worker, which must keep appending its own year's shared delta files.
        import shutil
        if os.path.abspath(out_dir) in (os.path.abspath(os.sep), os.path.abspath(".")):
            raise SystemExit(f"refusing to clear unsafe output_dir={out_dir!r}")
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        sample = select_sample(cfg)
        if args.limit:
            sample = sample[: args.limit]
        by_year: "OrderedDict[str, list]" = OrderedDict()
        for (yr, _relpath, inst) in sample:
            by_year.setdefault(yr, []).append(inst["hash_id"])

        print(f"DRIVER: {len(sample)} instances across {list(by_year)} -> ONE SUBPROCESS PER YEAR "
              f"(peak RAM = 1 year's index; full OS reclaim between years).", flush=True)
        tmpdir = tempfile.mkdtemp(prefix="gather_cfg_")
        n_years = len(by_year)
        for idx, (yr, ids) in enumerate(by_year.items(), 1):
            child_cfg = {**cfg, "sample": {**cfg["sample"], "years": [yr],
                                           "hash_ids": ids, "n_instances": len(ids)}}
            cfg_path = os.path.join(tmpdir, f"cfg_{yr}.json")
            with open(cfg_path, "w") as f:
                json.dump(child_cfg, f)
            bar = "=" * 78
            print(f"\n{bar}\n>>> SUBPROCESS {idx}/{n_years}: year {yr} ({len(ids)} instances) "
                  f"— fresh process; its RAM is fully reclaimed on exit\n{bar}", flush=True)
            rc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--config", cfg_path, "--single"]
            ).returncode
            if rc != 0:
                print(f"!!! year {yr} subprocess FAILED (rc={rc}) — aborting the driver.", flush=True)
                sys.exit(rc)

        _merge_inspect(out_dir)
        n = sum(1 for fp in glob.glob(os.path.join(out_dir, "snapshot_*", "delta_*.jsonl"))
                for _ in open(fp))
        print(f"\n=== ALL DONE (subprocess-per-year). {n} component files + inspect.txt under {out_dir} ===",
              flush=True)
        return

    # WORKER: single-year config, a `--single` child, or forced in-process (`--no-subprocess`).
    n_written = gather_worker(cfg, args.limit)
    _merge_inspect(out_dir)
    if not args.single:   # the driver prints the final ALL DONE for the multi-year subprocess path
        print(f"\n=== ALL DONE. {n_written} component files + inspect.txt under {out_dir} ===", flush=True)


if __name__ == "__main__":
    main()
