#!/usr/bin/env python3
"""Dataset-integrity audit: verify the tkgu_triples of a dataset (evaluation set or corpus) are CONSISTENT with the
KG snapshots. Run once as QA after the last construction step; HARD-FAIL on any violation; never modifies anything.

The rules are the ones the construction applies (src/dataset/emerge/tkgu_rules.py, manuscript §3), per operation:

ENTITY level ("in KG" = the entity appears in >= 1 snapshot triple OR in the snapshot's entity dictionary):
  * the emerging flags must fit the op (Mint+Add >= 1 emerging, Infer exactly 1, Exists / Add none) -> FLAG-STRUCTURE
  * existing (non-emerging) entity ends  -> MUST be in the KG       -> else EXISTING-ENT-MISSING
  * minted / emerging entity ends        -> MUST NOT be in the KG   -> else MINTED-ENT-PRESENT
  * relation type (P-id) not used in any snapshot triple -> NOTE RELATION-UNUSED (reported, NOT a failure): the TKGU
                                            rules put no condition on the relation (a property may be new in
                                            Wikidata, or used only outside the snapshot's entities); it only tells
                                            that a snapshot-scoped relation index would not contain it

TRIPLE level (per gold triple, keyed by `tkgu_operations`):
  * x-triples (Exists) -> triple (h,p,t) MUST be present in the snapshot                    -> else TRIPLE-MISSING
  * e-triples (Add) / ee-triples (Mint+Add) / ee-kg-triples (Infer) -> triple MUST be ABSENT -> else TRIPLE-PRESENT
  * d-triples (Deprecate) -> NO rule: a text may deprecate a fact that is or is not in the snapshot, also one of
    an entity being minted (the paper's own example). A triple whose only op is Deprecate is not checked at all
    (entity, relation or triple level); requiring presence was a bug (it removed correct gold triples).
  * a triple whose ops imply BOTH present and absent is itself a labeling contradiction      -> OP-CONTRADICTION

This is the triple-level companion to `filter_indices_to_snapshot.py`'s id-level checks: it guarantees
that scoping the indices to snapshot ids can never silently drop a retrievable gold, and that the op
labels match snapshot membership.

Memory-light: reads the dataset FIRST (collects the gold triples / entities / relations of a year), then streams the
snapshot `.gz` and the entity dictionary `.gz` once each, keeping only the wanted keys. Plain timestamped progress
lines every 30 s.

USAGE:
  PYTHONPATH=src python scripts/kg_context/check_snapshot_consistency.py \
    --snapshot-dir     /.../kg_snapshots \
    --entity-dicts-dir /.../indices/relik_entity_index \
    --eval-set-dir     /.../data/evaluation_set \
    --years 2019 2020 2021 2022 2023 2024 2025 \
    [--snapshot-template 'kg_snapshot_{year}-01-01.tsv.gz'] [--max-examples 10] [--warn-only]
  (--eval-set-dir may also point at the corpus.)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
import os
import time

from dataset.emerge import tkgu_rules as R

PRESENT_OPS = {R.EXISTS}                                       # triple must be IN the snapshot
UNCHECKED_OPS = {R.DEPRECATE}                                  # Deprecate: no consistency rule
ABSENT_OPS = {R.ADD, R.MINT_ADD, R.INFER}                      # triple must NOT be in the snapshot
OP_LABEL = {R.EXISTS: "exists", R.ADD: "add", R.MINT_ADD: "mint_add", R.INFER: "infer", R.DEPRECATE: "deprecate"}
ENTITY_KIND = {R.FLAG_STRUCTURE: "FLAG-STRUCTURE", R.MINTED_IN_KG: "MINTED-ENT-PRESENT",
               R.EXISTING_NOT_IN_KG: "EXISTING-ENT-MISSING"}
TRIPLE_KIND = {R.NOT_IN_T: "TRIPLE-MISSING", R.IN_T: "TRIPLE-PRESENT"}
PROGRESS_SECS = 30
T0 = time.time()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s]", *a, flush=True)


def collect_gold(eval_dir: str, year: str) -> tuple[list[dict], int]:
    """(all gold triples of a year: {h,p,t, emerging_head, emerging_tail, ops, hash_id, labels}, number of files).
    A year's files are `snapshot_<year>-01-01/delta_*.jsonl` (matched on the path below `eval_dir`)."""
    out = []
    files = sorted(glob.glob(os.path.join(eval_dir, f"snapshot_{year}-01-01", "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no snapshot_{year}-01-01/*.jsonl under {eval_dir}")
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                inst = json.loads(line)
                for gt in inst["tkgu_triples"]:
                    tr = gt["triple"]
                    if len(tr) != 3:
                        raise ValueError(f"{fp}: triple of {inst['hash_id']} does not have 3 elements: {tr!r}")
                    h, p, t = (str(x) for x in tr)
                    out.append({"h": h, "p": p, "t": t,
                                "eh": gt["emerging_head"], "et": gt["emerging_tail"],
                                "ops": list(gt["tkgu_operations"]),
                                "hash_id": inst["hash_id"],
                                # display only (violation examples)
                                "labels": gt["triple_labels"] if "triple_labels" in gt else None})
    return out, len(files)


def resolve_snapshot(path: str) -> str | None:
    """On-disk snapshot to read: the streamed `.gz` if present, else a plain `.tsv`, else None.
    (Same rule as src/kg_context/snapshot_io.py.)"""
    gz_path = path if path.endswith(".gz") else path + ".gz"
    if os.path.exists(gz_path):
        return gz_path
    if not path.endswith(".gz") and os.path.exists(path):
        return path
    return None


def scan_snapshot(path: str, want_triples: set[tuple], want_entities: set[str],
                  want_relations: set[str]) -> tuple[set, set, set]:
    """One pass, wanted keys only: (entities in some triple, relations used, found triples). Streams a `.gz`."""
    qids: set[str] = set()
    pids: set[str] = set()
    found: set[tuple] = set()
    log(f"    scanning {os.path.basename(path)} ({os.path.getsize(path)/1e9:.1f} GB on disk) ...")
    opener = gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")
    start = last = time.time()
    nr = 0
    with opener as f:
        for nr, line in enumerate(f, start=1):
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if len(parts) < 3:
                raise ValueError(f"{path}:{nr}: expected >= 3 TAB-separated fields, got {len(parts)}")
            h, p, t = parts[0], parts[1], parts[2]
            if h in want_entities:
                qids.add(h)
            if t in want_entities:
                qids.add(t)
            if p in want_relations:
                pids.add(p)
            if (h, p, t) in want_triples:
                found.add((h, p, t))
            if time.time() - last >= PROGRESS_SECS:
                log(f"    {os.path.basename(path)}: {nr:,} lines, {len(found):,}/{len(want_triples):,} gold triples found")
                last = time.time()
    log(f"    {os.path.basename(path)}: {nr:,} lines in {time.time() - start:.0f}s")
    return qids, pids, found


def audit_year(snapshot: str, entity_dicts_dir: str, eval_dir: str, year: str,
               max_ex: int) -> tuple[list[str], dict]:
    gold, nfiles = collect_gold(eval_dir, year)
    log(f"    {len(gold):,} gold triples from {nfiles} file(s)")
    want = {(g["h"], g["p"], g["t"]) for g in gold}
    want_entities = {g["h"] for g in gold} | {g["t"] for g in gold}
    qids, pids, found = scan_snapshot(snapshot, want, want_entities, {g["p"] for g in gold})
    in_dictionary = R.scan_entity_dictionary(R.entity_dictionary_file(entity_dicts_dir, f"{year}-01-01"),
                                             want_entities)
    log(f"    snapshot: {len(found):,}/{len(want):,} distinct gold triples present; {len(qids):,} gold entities in "
        f"some triple, {len(in_dictionary):,} in the entity dictionary, {len(want_entities):,} distinct")

    def in_kg(entity: str) -> bool:
        return entity in qids or entity in in_dictionary

    viol: list[str] = []
    fails: dict[str, int] = {}
    counts = {lab: {"n": 0} for lab in OP_LABEL.values()}

    def add(kind: str, g: dict, detail: str):
        fails[kind] = (fails[kind] if kind in fails else 0) + 1   # count ALL violations (not just the capped examples)
        if fails[kind] <= max_ex:
            viol.append(f"{kind} [{year}] {detail} | triple={g['labels']} ({g['h']},{g['p']},{g['t']}) "
                        f"ops={sorted(g['ops'])} hash={g['hash_id']}")

    notes: dict[str, int] = {}
    note_lines: list[str] = []

    def add_note(kind: str, g: dict, detail: str):
        notes[kind] = (notes[kind] if kind in notes else 0) + 1
        if notes[kind] <= max_ex:
            note_lines.append(f"{kind} [{year}] {detail} | triple={g['labels']} ({g['h']},{g['p']},{g['t']}) "
                              f"ops={sorted(g['ops'])} hash={g['hash_id']}")

    for g in gold:
        ops = set(g["ops"])
        for op in ops:
            counts[OP_LABEL[op]]["n"] += 1                        # unknown op -> KeyError (fail loudly)
        if ops and ops <= UNCHECKED_OPS:
            continue  # Deprecate-only triple: no rule (see module docstring)
        is_present = (g["h"], g["p"], g["t"]) in found
        checked = sorted(ops - UNCHECKED_OPS)
        verdicts = {op: R.op_violation(op, is_present, g["eh"], g["et"], in_kg(g["h"]), in_kg(g["t"]))
                    for op in checked}
        # entity presence + flags (once per kind and triple)
        for kind in sorted({ENTITY_KIND[er] for _, er in verdicts.values() if er is not None}):
            add(kind, g, f"entity rule ({kind.lower()}); head emerging={g['eh']} in KG={in_kg(g['h'])}, "
                         f"tail emerging={g['et']} in KG={in_kg(g['t'])}")
        # relation presence: a note only (no TKGU rule on the relation)
        if g["p"] not in pids:
            add_note("RELATION-UNUSED", g, f"relation {g['p']} not used in snapshot")
        # triple presence by op
        if ops & PRESENT_OPS and ops & ABSENT_OPS:
            add("OP-CONTRADICTION", g, "ops imply both present AND absent")
            continue
        for kind in sorted({TRIPLE_KIND[tr] for tr, _ in verdicts.values() if tr is not None}):
            add(kind, g, "exists triple NOT in snapshot" if kind == "TRIPLE-MISSING"
                else "add/mint+add/infer triple IS in snapshot")

    n_fail = sum(fails.values())
    return viol, {"gold": len(gold), "distinct_triples": len(want), "op_counts": counts,
                  "n_fail": n_fail, "fails": fails, "notes": notes, "note_lines": note_lines}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--snapshot-template", default="kg_snapshot_{year}-01-01.tsv.gz")
    ap.add_argument("--entity-dicts-dir", required=True,
                    help="per-snapshot entity dictionaries <dir>/<year>-01-01/documents.jsonl.gz (\"in KG\" = in a "
                         "snapshot triple OR in this dictionary)")
    ap.add_argument("--eval-set-dir", required=True, help="evaluation set or corpus (snapshot_*/delta_*.jsonl)")
    ap.add_argument("--years", nargs="+", required=True)
    ap.add_argument("--max-examples", type=int, default=10, help="max example violations to print per kind/year")
    ap.add_argument("--warn-only", action="store_true", help="report but do NOT hard-fail (exit 0)")
    args = ap.parse_args()
    # tkgu_rules logs its entity-dictionary progress through `logging`
    logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

    total_viol = 0
    summary = []
    for year in args.years:
        log(f"===================== YEAR {year} =====================")
        configured = os.path.join(args.snapshot_dir, args.snapshot_template.format(year=year))
        snap = resolve_snapshot(configured)
        if snap is None:
            raise SystemExit(f"FATAL: snapshot missing: {configured} (nor {configured}.gz)")
        viol, stats = audit_year(snap, args.entity_dicts_dir, args.eval_set_dir, year, args.max_examples)
        summary.append((year, stats))
        if stats["n_fail"]:
            rate = 100 * stats["n_fail"] / stats["gold"] if stats["gold"] else 0
            log(f"    !!! {stats['n_fail']:,} violation(s) ({rate:.2f}% of {stats['gold']:,} gold) "
                f"by kind: {dict(stats['fails'])}; showing up to {args.max_examples}/kind:")
            for v in viol:
                log(f"      {v}")
        else:
            log("    OK: no inconsistencies")
        if stats["notes"]:
            log(f"    note (not a failure): {dict(stats['notes'])}; showing up to {args.max_examples}/kind:")
            for n in stats["note_lines"]:
                log(f"      {n}")
        total_viol += stats["n_fail"]

    log("===================== AUDIT SUMMARY =====================")
    log(f"{'year':6} {'gold':>8} {'distinct':>9} {'violations':>11} {'fail%':>7} {'notes':>6}   op-counts")
    for year, stats in summary:
        opc = " ".join(f"{lab}={stats['op_counts'][lab]['n']}" for lab in OP_LABEL.values())
        rate = 100 * stats["n_fail"] / stats["gold"] if stats["gold"] else 0
        log(f"{year:6} {stats['gold']:>8,} {stats['distinct_triples']:>9,} {stats['n_fail']:>11,} {rate:>6.2f}% "
            f"{sum(stats['notes'].values()):>6,}   {opc}")

    if total_viol:
        msg = f"{total_viol} violation(s) across years -- dataset/snapshot INCONSISTENT, investigate."
        if args.warn_only:
            log("WARN-ONLY: " + msg)
        else:
            raise SystemExit("FATAL: " + msg)
    else:
        log("ALL YEARS CONSISTENT: entity/relation/triple presence matches the snapshots.")


if __name__ == "__main__":
    main()
