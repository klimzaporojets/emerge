#!/usr/bin/env python3
"""Filter relik entity/relation indices to the ids EFFECTIVELY PRESENT in the KG snapshots.

WHY
---
The s08 relik indices are built over the FULL vocabularies:
  * relation index  : ~10.6k Wikidata properties, of which ~69% are external-ID properties
                      (VIAF/GND/IMDb/... IDs) that NEVER appear as edges in the KG snapshots.
  * entity index    : ~6.67M Wikipedia entities per year.
The KG snapshots are ENTITY-ENTITY graphs that use only ~1k relation types and a subset of the
entities. Every index row whose id is not in the snapshot is a pure retrieval DISTRACTOR: for the
relation pool the ~7.4k external-ID properties crowd out the true relations and drag relation
recall down badly (measured: relation recall@8 ~59%). This tool produces NEW, SNAPSHOT-SCOPED
indices that keep ONLY the ids present in each year's snapshot.

HOW
---
Pure CPU row-slicing of the row-aligned `embeddings.pt` -- NO re-embedding, NO GPU. Never touches
the source; writes a fresh tree under --out-base. Alignment invariant (verified for these indices):
`embeddings.pt` is a [N, dim] tensor whose row i corresponds to line i of `documents.jsonl`. AIDA
reuses the native entity `documents.jsonl` (symlink) with its OWN `embeddings.pt` at the SAME
alignment, so the kept-row set computed from the native entity documents applies to BOTH native and
AIDA embeddings. The script HARD-FAILS if that invariant does not hold (row count mismatch).

OUTPUT LAYOUT (mirrors the source so augment configs point straight at <out-base>/<basename>):
  <out-base>/<relation-basename>/<year>/{documents.jsonl, embeddings.pt, config.yaml}
  <out-base>/<entity-native-basename>/<year>/{documents.jsonl, embeddings.pt, config.yaml[, config_emerge.yaml]}
  <out-base>/<entity-aida-basename>/<year>/{documents.jsonl -> ../../<nat-basename>/<year>/documents.jsonl, embeddings.pt, config.yaml}
Plus <out-base>/filter_stats.json and <out-base>/README.md (auto-generated with the real numbers).

USAGE (run once; CPU). Preview reductions cheaply first with --dry-run (no embeddings load/write):
  PYTHONPATH=src python scripts/kg_context/filter_indices_to_snapshot.py \
    --relation-index-base   /.../indexes-e5-small-v2-wikipedia-relations \
    --entity-native-base    /.../indexes-e5-small-v2-wikipedia-matryoshka \
    --entity-aida-base      /.../relik-retriever-e5-base-v2-aida-blink-encoder \
    --snapshot-dir          /.../kg_snapshots \
    --snapshot-template     'kg_snapshot_{year}-01-01.tsv.gz' \
    --out-base              /.../s08_snapshot_scoped \
    --years 2019 2020 2021 2022 2023 2024 2025 \
    [--pools relation entity_native entity_aida] [--mmap] [--dry-run] \
    [--check-gold-eval-set /.../data/evaluation_set]   # optional hard-fail integrity audit

Peak RAM ~= one entity embeddings.pt (~10 GB) unless --mmap (reads only kept rows from disk).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import shutil
import sys
import time

try:
    from tqdm import tqdm
except Exception:  # tqdm always present in the relik env; degrade gracefully if not
    def tqdm(it, **_):
        return it

POOLS = ("relation", "entity_native", "entity_aida")
META_KEY = {"relation": "property", "entity_native": "wikidata", "entity_aida": "wikidata"}


def log(*a):
    print(*a, flush=True)


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024


def gold_violation(warn_only: bool, msg: str) -> None:
    """Report a gold-integrity violation. With --warn-only: log + keep going (counts reported at the
    end); else HARD-FAIL. Structural errors (missing files, tensor/alignment) always hard-fail regardless."""
    if warn_only:
        log("    ⚠ GOLD-CHECK VIOLATION: " + msg)
    else:
        raise SystemExit("FATAL gold-check: " + msg + "  (pass --warn-only to report instead of stop)")


# --------------------------------------------------------------------------- snapshots

def resolve_snapshot(path: str) -> str | None:
    """On-disk snapshot to read: the streamed `.gz` if present, else a plain `.tsv`, else None.
    (Same rule as src/kg_context/snapshot_io.py; inlined to keep this script standalone.)"""
    gz_path = path if path.endswith(".gz") else path + ".gz"
    if os.path.exists(gz_path):
        return gz_path
    if not path.endswith(".gz") and os.path.exists(path):
        return path
    return None


def scan_snapshot(path: str, want_qids: bool, want_pids: bool) -> tuple[set, set]:
    """One pass over a snapshot TSV (head_qid<TAB>pid<TAB>tail_qid<TAB>...). Returns (qids, pids).
    Streams a `.gz` directly (never decompress)."""
    qids: set[str] = set()
    pids: set[str] = set()
    n = 0
    sz = os.path.getsize(path)
    log(f"    loading snapshot {os.path.basename(path)} ({human(sz)} on disk) ...")
    opener = gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")
    with opener as f:
        for line in tqdm(f, desc=f"    scan {os.path.basename(path)}", unit="edge", unit_scale=True):
            parts = line.split("\t", 3)
            if len(parts) < 3:
                continue
            n += 1
            if want_qids:
                qids.add(parts[0])
                qids.add(parts[2])
            if want_pids:
                pids.add(parts[1])
    log(f"    snapshot: {n:,} edges -> {len(qids):,} distinct entity QIDs, {len(pids):,} distinct relation PIDs")
    return qids, pids


def gold_ids_from_eval_set(eval_dir: str, year: str) -> tuple[set, set]:
    """INTEGRITY CHECK source: existing (non-emerging) gold entity QIDs / relation PIDs for a year,
    read from the raw evaluation set (tkgu_triples). These are NOT added to the allow-set -- they
    are used only to VERIFY (and HARD-FAIL) that every existing gold id is (a) present in the KG
    snapshot and (b) present in the index dictionary. We never complete/fabricate missing rows.
    Returns (existing_qids, existing_pids, minted_qids):
      existing_* = non-emerging ends / relations that MUST be in the snapshot (and index).
      minted_qids = emerging ends that MUST NOT be in the snapshot (created after snapshot date).
    Best-effort over the eval-set schema."""
    exist_q: set[str] = set()
    exist_p: set[str] = set()
    mint_q: set[str] = set()
    files = glob.glob(os.path.join(eval_dir, "**", "*.jsonl"), recursive=True)
    for fp in files:
        # only files whose path mentions the year (snapshot_<year> or year in name) to stay cheap
        if year not in fp:
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    inst = json.loads(line)
                except Exception:
                    continue
                for t in inst.get("tkgu_triples", []) or []:
                    tr = t.get("triple") or []       # [head_qid, rel_pid, tail_qid]
                    if len(tr) != 3:
                        continue
                    h, p, tl = tr
                    if h:
                        (mint_q if t.get("emerging_head") else exist_q).add(h)
                    if tl:
                        (mint_q if t.get("emerging_tail") else exist_q).add(tl)
                    if p:
                        exist_p.add(p)     # relation types are never "minted"; must be a used snapshot relation
    return exist_q, exist_p, mint_q


# --------------------------------------------------------------------------- one index dir

def scan_and_write(documents_path: str, allow: set, meta_key: str, dst_path: str | None,
                   collect_all: bool = False) -> tuple[list[int], int, set | None]:
    """SINGLE pass over documents.jsonl: decide kept rows AND (if dst_path) write the filtered,
    id-renumbered documents -- so the 6.67M-line entity dictionary is parsed ONCE, not twice.
    Row ordinal (0-based doc count) == embeddings.pt row (robust to absent blank lines).
    Returns (kept_ordinals, total_rows, all_ids_or_None). all_ids only when collect_all (gold-in-index)."""
    kept: list[int] = []
    ordinal = 0
    new_id = 0
    full: set | None = set() if collect_all else None
    log(f"    {'scan+write' if dst_path else 'scan'} "
        f"{os.path.basename(os.path.dirname(documents_path))}/documents.jsonl "
        f"({human(os.path.getsize(documents_path))}) ...")
    fout = open(dst_path, "w", encoding="utf-8") if dst_path else None
    try:
        with open(documents_path, encoding="utf-8") as fin:
            for line in tqdm(fin, desc="    docs", unit="doc", unit_scale=True):
                s = line.strip()
                if not s:
                    continue
                d = json.loads(s)
                cid = (d.get("metadata") or {}).get(meta_key)
                if collect_all and cid is not None:
                    full.add(cid)
                if cid is not None and cid in allow:
                    kept.append(ordinal)
                    if fout is not None:
                        d["id"] = new_id
                        new_id += 1
                        fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                ordinal += 1
    finally:
        if fout is not None:
            fout.close()
    if fout is not None:
        assert new_id == len(kept), f"wrote {new_id} docs, expected {len(kept)}"
    return kept, ordinal, full


# (doc writing is merged into scan_and_write above: the filtered, id-renumbered documents are written in
#  the SAME pass that decides the kept rows, so the 6.67M-line entity dictionary is parsed only once.)


def slice_embeddings(src_path: str, dst_path: str, kept: list[int], total_docs: int, mmap: bool) -> tuple[int, int]:
    """torch.load the [N, dim] tensor, HARD-CHECK N == total_docs, index by kept rows, torch.save.
    Returns (in_bytes, out_bytes)."""
    import torch
    log(f"    loading embeddings.pt ({human(os.path.getsize(src_path))}{', mmap' if mmap else ''}) ... "
        f"[this is the slow step]")
    t = time.time()
    try:
        emb = torch.load(src_path, map_location="cpu", mmap=mmap)
    except TypeError:  # torch too old for mmap kwarg
        emb = torch.load(src_path, map_location="cpu")
    log(f"      loaded in {time.time()-t:.0f}s")
    if not torch.is_tensor(emb):
        # tolerate a dict/list container holding exactly one tensor; else fail loudly (no silent guess)
        cand = None
        if isinstance(emb, dict):
            tensors = [v for v in emb.values() if torch.is_tensor(v)]
            cand = tensors[0] if len(tensors) == 1 else None
        elif isinstance(emb, (list, tuple)) and len(emb) == 1 and torch.is_tensor(emb[0]):
            cand = emb[0]
        if cand is None:
            raise SystemExit(f"FATAL: {src_path} is a {type(emb)}, not a single tensor -- refuse to guess layout")
        emb = cand
    if emb.shape[0] != total_docs:
        raise SystemExit(f"FATAL alignment: {src_path} has {emb.shape[0]} rows but documents.jsonl has "
                         f"{total_docs} lines -- row i must map to doc i. Refusing to slice.")
    log(f"    slicing {len(kept):,} of {emb.shape[0]:,} rows (dim={emb.shape[1]}, {emb.dtype}) ...")
    idx = torch.as_tensor(kept, dtype=torch.long)
    out = emb.index_select(0, idx).contiguous().clone()
    log(f"    saving embeddings.pt -> {os.path.basename(os.path.dirname(dst_path))}/ ...")
    torch.save(out, dst_path)
    return os.path.getsize(src_path), os.path.getsize(dst_path)


def copy_configs(src_dir: str, dst_dir: str) -> None:
    for name in ("config.yaml", "config_emerge.yaml"):
        s = os.path.join(src_dir, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst_dir, name))


# --------------------------------------------------------------------------- driver

def process_pool(pool: str, src_base: str, out_base: str, year: str, allow: set,
                 dry_run: bool, mmap: bool, nat_out_basename: str, gold_ids: set | None = None,
                 warn_only: bool = False, audit: dict | None = None,
                 reuse: tuple | None = None) -> tuple[dict, tuple]:
    date = f"{year}-01-01"
    src_dir = os.path.join(src_base, date)
    if not os.path.isdir(src_dir):
        raise SystemExit(f"FATAL: source index dir missing: {src_dir}")
    documents = os.path.join(src_dir, "documents.jsonl")  # follows symlink transparently for AIDA
    if not os.path.isfile(documents):
        raise SystemExit(f"FATAL: {documents} not found")

    meta_key = META_KEY[pool]
    out_basename = os.path.basename(src_base.rstrip("/"))
    dst_dir = os.path.join(out_base, out_basename, date)
    if not dry_run:
        os.makedirs(dst_dir, exist_ok=True)
        copy_configs(src_dir, dst_dir)

    if reuse is not None:
        # AIDA shares native's documents.jsonl (symlink) -> reuse native's kept/total/full_ids; slice emb only.
        kept, total, full_ids = reuse
        if not dry_run:
            rel = os.path.relpath(os.path.join(out_base, nat_out_basename, date, "documents.jsonl"), dst_dir)
            link = os.path.join(dst_dir, "documents.jsonl")
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(rel, link)
    else:
        dst_docs = None if dry_run else os.path.join(dst_dir, "documents.jsonl")
        kept, total, full_ids = scan_and_write(documents, allow, meta_key, dst_docs,
                                               collect_all=gold_ids is not None)
        if gold_ids is not None and full_ids is not None:
            missing = gold_ids - full_ids
            if audit is not None:
                audit["idx_checked"] += len(gold_ids)
                audit["idx_miss"] += len(missing)
            if missing:
                gold_violation(warn_only, f"[{pool} {year}] {len(missing)}/{len(gold_ids)} gold id(s) absent from "
                               f"the index dictionary -> unretrievable even before filtering. "
                               f"examples: {sorted(missing)[:10]}")
            else:
                log(f"    [{pool}] gold-in-index check OK ({len(gold_ids):,} gold ids all present)")

    stat = {"pool": pool, "year": year, "total_rows": total, "kept_rows": len(kept),
            "dropped_rows": total - len(kept),
            "pct_reduced": round(100 * (total - len(kept)) / total, 2) if total else 0.0}
    if not dry_run:
        in_b, out_b = slice_embeddings(os.path.join(src_dir, "embeddings.pt"),
                                       os.path.join(dst_dir, "embeddings.pt"), kept, total, mmap)
        stat["emb_in_bytes"], stat["emb_out_bytes"] = in_b, out_b
    log(f"    [{pool}] {year}: kept {len(kept):,}/{total:,} "
        f"({stat['pct_reduced']:.1f}% dropped)"
        + (f"  emb {human(stat.get('emb_in_bytes',0))} -> {human(stat.get('emb_out_bytes',0))}" if not dry_run else "  (dry-run)"))
    return stat, (kept, total, full_ids)


def process_year(year: str, ctx: dict) -> tuple[list[dict], dict]:
    """All work for one year (snapshot scan + optional gold checks + each pool). Self-contained and
    picklable-arg'd so it can run in a worker process (year-level parallelism). Returns (stats, audit_delta)."""
    log(f"\n===================== YEAR {year} =====================")
    configured = os.path.join(ctx["snapshot_dir"], ctx["snapshot_template"].format(year=year))
    snap = resolve_snapshot(configured)
    if snap is None:
        raise SystemExit(f"FATAL: snapshot missing: {configured} (nor {configured}.gz)")
    want_qids, want_pids, warn_only = ctx["want_qids"], ctx["want_pids"], ctx["warn_only"]
    qids, pids = scan_snapshot(snap, want_qids, want_pids)
    audit = {"exist_checked": 0, "exist_miss": 0, "mint_checked": 0, "mint_present": 0,
             "idx_checked": 0, "idx_miss": 0}
    gold_exist_q = gold_exist_p = None
    if ctx["check_gold_eval_set"]:
        gq, gp, mq = gold_ids_from_eval_set(ctx["check_gold_eval_set"], year)
        if not (gq or gp or mq):
            log(f"    WARNING: --check-gold-eval-set gave 0 gold ids for {year} (no eval instances matched under "
                f"{ctx['check_gold_eval_set']}); check is a NO-OP this year")
        miss_q = (gq - qids) if want_qids else set()
        miss_p = (gp - pids) if want_pids else set()
        audit["exist_checked"] += (len(gq) if want_qids else 0) + (len(gp) if want_pids else 0)
        audit["exist_miss"] += len(miss_q) + len(miss_p)
        if miss_q or miss_p:
            gold_violation(warn_only, f"[{year}] {len(miss_q)} existing-entity + {len(miss_p)} relation gold id(s) "
                           f"NOT in the snapshot (supposedly-existing gold absent from the KG). "
                           f"examples ent={sorted(miss_q)[:5]} rel={sorted(miss_p)[:5]}")
        present = (mq & qids) if want_qids else set()
        audit["mint_checked"] += len(mq) if want_qids else 0
        audit["mint_present"] += len(present)
        if present:
            gold_violation(warn_only, f"[{year}] {len(present)} emerging entity id(s) ARE in the snapshot "
                           f"(should exist only after the snapshot date). examples {sorted(present)[:10]}")
        gold_exist_q, gold_exist_p = gq, gp
        log(f"    gold [{year}]: existing-miss={len(miss_q)+len(miss_p)}"
            f"/{(len(gq) if want_qids else 0)+(len(gp) if want_pids else 0)}, "
            f"minted-present={len(present)}/{len(mq) if want_qids else 0}")
    allow = {"relation": pids, "entity_native": qids, "entity_aida": qids}
    gold_for = {"relation": gold_exist_p, "entity_native": gold_exist_q, "entity_aida": gold_exist_q}
    stats: list[dict] = []
    nat_reuse = None
    for pool in [p for p in POOLS if p in ctx["pools"]]:   # native before aida (aida reuses native's kept-rows)
        stat, kt = process_pool(pool, ctx["base_of"][pool], ctx["out_base"], year, allow[pool],
                                ctx["dry_run"], ctx["mmap"], ctx["nat_out_basename"], gold_ids=gold_for[pool],
                                warn_only=warn_only, audit=audit,
                                reuse=(nat_reuse if pool == "entity_aida" else None))
        stats.append(stat)
        if pool == "entity_native":
            nat_reuse = kt
    return stats, audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relation-index-base")
    ap.add_argument("--entity-native-base")
    ap.add_argument("--entity-aida-base")
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--snapshot-template", default="kg_snapshot_{year}-01-01.tsv.gz")
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--years", nargs="+", required=True)
    ap.add_argument("--pools", nargs="+", choices=POOLS, default=list(POOLS))
    ap.add_argument("--check-gold-eval-set", default=None,
                    help="optional integrity audit (never modifies output): HARD-FAIL if any existing gold id "
                         "from this eval-set dir is missing from the snapshot or the index dictionary, or if any "
                         "minted/emerging entity is present in the snapshot.")
    ap.add_argument("--mmap", action="store_true", help="mmap embeddings.pt (lower peak RAM; needs recent torch)")
    ap.add_argument("--dry-run", action="store_true", help="scan + count reductions only; no embeddings load/write")
    ap.add_argument("--warn-only", action="store_true",
                    help="report gold-check violations + percentages and KEEP GOING, instead of hard-failing "
                         "(structural errors -- missing files, tensor/alignment -- still hard-fail). For sbatch runs.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="process this many YEARS in parallel (worker processes). Peak RAM ~= jobs x ~12 GB "
                         "(one entity emb slice at a time per worker). Use with --warn-only for parallel runs.")
    args = ap.parse_args()

    pools = list(args.pools)
    base_of = {"relation": args.relation_index_base, "entity_native": args.entity_native_base,
               "entity_aida": args.entity_aida_base}
    for p in pools:
        if not base_of[p]:
            raise SystemExit(f"FATAL: --{p.replace('_','-')}-base is required for pool '{p}'")
    if "entity_aida" in pools and "entity_native" not in pools:
        raise SystemExit("FATAL: entity_aida shares the native documents.jsonl -> include entity_native too")
    nat_out_basename = os.path.basename((args.entity_native_base or "").rstrip("/"))

    want_qids = any(p.startswith("entity") for p in pools)
    want_pids = "relation" in pools

    if not args.dry_run:
        os.makedirs(args.out_base, exist_ok=True)

    ctx = {"snapshot_dir": args.snapshot_dir, "snapshot_template": args.snapshot_template,
           "out_base": args.out_base, "pools": pools, "base_of": base_of,
           "want_qids": want_qids, "want_pids": want_pids, "nat_out_basename": nat_out_basename,
           "check_gold_eval_set": args.check_gold_eval_set, "dry_run": args.dry_run,
           "mmap": args.mmap, "warn_only": args.warn_only}

    all_stats: list[dict] = []
    audit = {"exist_checked": 0, "exist_miss": 0, "mint_checked": 0, "mint_present": 0,
             "idx_checked": 0, "idx_miss": 0}
    t0 = time.time()
    jobs = max(1, min(args.jobs, len(args.years)))
    if jobs > 1:
        log(f"PARALLEL: {jobs} worker(s) over {len(args.years)} years")
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(process_year, args.years, [ctx] * len(args.years)))
    else:
        results = [process_year(year, ctx) for year in args.years]
    for stats, adelta in results:
        all_stats.extend(stats)
        for k in audit:
            audit[k] += adelta[k]

    # ---- stats summary + artifacts
    log("\n\n===================== REDUCTION SUMMARY =====================")
    log(f"{'pool':14} {'year':6} {'total':>12} {'kept':>12} {'%dropped':>9}")
    for s in all_stats:
        log(f"{s['pool']:14} {s['year']:6} {s['total_rows']:>12,} {s['kept_rows']:>12,} {s['pct_reduced']:>8.1f}%")

    if args.check_gold_eval_set:
        def _pct(n, d):
            return f"{100 * n / d:.3f}%" if d else "n/a"
        log("\n===================== GOLD-CHECK AUDIT (report-only) =====================")
        log(f"  existing gold NOT in snapshot : {audit['exist_miss']:,}/{audit['exist_checked']:,} "
            f"({_pct(audit['exist_miss'], audit['exist_checked'])})   [expected 0]")
        log(f"  minted gold IS in snapshot    : {audit['mint_present']:,}/{audit['mint_checked']:,} "
            f"({_pct(audit['mint_present'], audit['mint_checked'])})   [expected 0]")
        log(f"  gold NOT in index dictionary  : {audit['idx_miss']:,}/{audit['idx_checked']:,} "
            f"({_pct(audit['idx_miss'], audit['idx_checked'])})   [expected 0]")
    log(f"\n[done in {time.time()-t0:.0f}s]")

    if not args.dry_run:
        with open(os.path.join(args.out_base, "filter_stats.json"), "w") as f:
            json.dump({"args": vars(args), "stats": all_stats}, f, indent=2)
        write_readme(args, all_stats)
        log(f"wrote {os.path.join(args.out_base, 'filter_stats.json')} and README.md")


def write_readme(args, stats: list[dict]) -> None:
    lines = [
        "# Snapshot-scoped relik indices",
        "",
        "Entity/relation relik indices filtered to the ids **effectively present in each year's KG",
        "snapshot** (entity-entity graph). Produced by `scripts/kg_context/filter_indices_to_snapshot.py`",
        "(pure CPU row-slicing of the row-aligned `embeddings.pt`; NO re-embedding, NO GPU).",
        "",
        "## Why",
        "The source s08 indices cover the FULL vocabularies (relation: ~10.6k Wikidata properties, ~69%",
        "external-ID props never used as edges; entity: ~6.67M/yr). Those unused rows are retrieval",
        "distractors that dragged relation recall down. Here they are removed per year.",
        "",
        "## Source indices",
        f"- relation      : `{args.relation_index_base}`",
        f"- entity native : `{args.entity_native_base}`",
        f"- entity aida   : `{args.entity_aida_base}`",
        f"- snapshots     : `{args.snapshot_dir}/{args.snapshot_template}`",
        f"- gold-check    : `{args.check_gold_eval_set or 'OFF'}`",
        "",
        "## Layout (point augment `*_index_base_path` at these)",
        "```",
        "<basename>/<year>-01-01/{documents.jsonl, embeddings.pt, config.yaml}",
        "```",
        "AIDA `documents.jsonl` is a symlink to the native filtered documents (same rows, own embeddings),",
        "mirroring the source.",
        "",
        "## Reduction stats",
        "| pool | year | total | kept | % dropped |",
        "|------|------|------:|-----:|----------:|",
    ]
    for s in stats:
        lines.append(f"| {s['pool']} | {s['year']} | {s['total_rows']:,} | {s['kept_rows']:,} | {s['pct_reduced']:.1f}% |")
    lines += ["", "## Provenance", "Never overwrites the source. Alignment invariant checked at build time",
              "(`embeddings.pt` row i == `documents.jsonl` line i; build hard-fails on mismatch)."]
    with open(os.path.join(args.out_base, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
