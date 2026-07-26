#!/usr/bin/env python3
"""Dataset-integrity audit: verify the eval-set tkgu_triples are CONSISTENT with the KG snapshots.

For each year, checks (HARD-FAIL on any violation; never modifies anything) at two levels:

ENTITY / RELATION level (per gold triple end):
  * existing (non-emerging) entity ends  -> MUST be in the snapshot
  * minted / emerging entity ends        -> MUST NOT be in the snapshot (created after the snapshot date)
  * relation type (P-id)                 -> MUST be in the snapshot (else the snapshot-scoped relation
                                            index would drop it -> gold unretrievable)

TRIPLE level (per gold triple, keyed by `tkgu_operations`):
  * x-triples (Exists) / d-triples (Deprecate) -> triple (h,p,t) MUST be present in the snapshot
  * e-triples (Add) / ee-triples (Mint+Add) / ee-kg-triples (Infer) -> triple MUST be ABSENT
  * a triple whose ops imply BOTH present and absent is itself a labeling contradiction -> fail

This is the triple-level companion to `filter_indices_to_snapshot.py`'s id-level checks: it guarantees
that scoping the indices to snapshot ids can never silently drop a retrievable gold, and that the op
labels match snapshot membership. Run once as QA.

Memory-light: reads the eval-set FIRST (collects the ~1-2k gold triples/year), then a SINGLE snapshot
pass builds the entity-QID set + relation-PID set and marks which gold triples are found -- so we never
materialise the full ~35M-edge set.

USAGE:
  PYTHONPATH=src python scripts/kg_context/check_snapshot_consistency.py \
    --snapshot-dir   /.../kg_snapshots \
    --eval-set-dir   /.../data/evaluation_set \
    --years 2019 2020 2021 2022 2023 2024 2025 \
    [--snapshot-template 'kg_snapshot_{year}-01-01.tsv'] [--max-examples 10] [--warn-only]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **_):
        return it

PRESENT_OPS = {"x-triples", "d-triples"}                       # triple must be IN the snapshot
ABSENT_OPS = {"e-triples", "ee-triples", "ee-kg-triples"}      # triple must NOT be in the snapshot
OP_LABEL = {"x-triples": "exists", "e-triples": "add", "ee-triples": "mint_add",
            "ee-kg-triples": "infer", "d-triples": "deprecate"}


def log(*a):
    print(*a, flush=True)


def collect_gold(eval_dir: str, year: str) -> list[dict]:
    """All gold triples for a year: {h,p,t, emerging_head, emerging_tail, ops, hash_id, labels}."""
    out = []
    files = [fp for fp in glob.glob(os.path.join(eval_dir, "**", "*.jsonl"), recursive=True) if year in fp]
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                inst = json.loads(line)
                for gt in inst.get("tkgu_triples", []) or []:
                    tr = gt.get("triple") or []
                    if len(tr) != 3:
                        continue
                    h, p, t = tr
                    out.append({"h": h, "p": p, "t": t,
                                "eh": bool(gt.get("emerging_head")), "et": bool(gt.get("emerging_tail")),
                                "ops": set(gt.get("tkgu_operations") or []),
                                "hash_id": inst.get("hash_id"), "labels": gt.get("triple_labels")})
    return out, len(files)


def scan_snapshot(path: str, want_triples: set[tuple]) -> tuple[set, set, set]:
    """One pass: return (qids, pids, found_triples subset of want_triples)."""
    qids: set[str] = set()
    pids: set[str] = set()
    found: set[tuple] = set()
    log(f"    scanning {os.path.basename(path)} ({os.path.getsize(path)/1e9:.1f} GB) ...")
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"    scan {os.path.basename(path)}", unit="edge", unit_scale=True):
            parts = line.split("\t", 3)
            if len(parts) < 3:
                continue
            h, p, t = parts[0], parts[1], parts[2]
            qids.add(h)
            qids.add(t)
            pids.add(p)
            tr = (h, p, t)
            if tr in want_triples:
                found.add(tr)
    return qids, pids, found


def audit_year(snapshot: str, eval_dir: str, year: str, max_ex: int) -> tuple[list[str], dict]:
    gold, nfiles = collect_gold(eval_dir, year)
    log(f"    {len(gold):,} gold triples from {nfiles} eval file(s)")
    want = {(g["h"], g["p"], g["t"]) for g in gold}
    qids, pids, found = scan_snapshot(snapshot, want)
    log(f"    snapshot: {len(qids):,} QIDs, {len(pids):,} PIDs; {len(found):,}/{len(want):,} distinct gold triples present")

    viol: list[str] = []
    fails: dict[str, int] = {}
    counts = {lab: {"n": 0} for lab in OP_LABEL.values()}

    def add(kind: str, g: dict, detail: str):
        fails[kind] = fails.get(kind, 0) + 1          # count ALL violations (not just the capped examples)
        if fails[kind] <= max_ex:
            viol.append(f"{kind} [{year}] {detail} | triple={g['labels']} ({g['h']},{g['p']},{g['t']}) "
                        f"ops={sorted(g['ops'])} hash={g['hash_id']}")

    for g in gold:
        for op in (g["ops"] & set(OP_LABEL)):
            counts[OP_LABEL[op]]["n"] += 1
        # entity presence
        if not g["eh"] and g["h"] and g["h"] not in qids:
            add("EXISTING-ENT-MISSING", g, f"existing head {g['h']} not in snapshot")
        if g["eh"] and g["h"] and g["h"] in qids:
            add("MINTED-ENT-PRESENT", g, f"emerging head {g['h']} IS in snapshot")
        if not g["et"] and g["t"] and g["t"] not in qids:
            add("EXISTING-ENT-MISSING", g, f"existing tail {g['t']} not in snapshot")
        if g["et"] and g["t"] and g["t"] in qids:
            add("MINTED-ENT-PRESENT", g, f"emerging tail {g['t']} IS in snapshot")
        # relation presence
        if g["p"] and g["p"] not in pids:
            add("RELATION-MISSING", g, f"relation {g['p']} not used in snapshot")
        # triple presence by op
        want_present = bool(g["ops"] & PRESENT_OPS)
        want_absent = bool(g["ops"] & ABSENT_OPS)
        if want_present and want_absent:
            add("OP-CONTRADICTION", g, "ops imply both present AND absent")
            continue
        is_present = (g["h"], g["p"], g["t"]) in found
        if want_present and not is_present:
            add("TRIPLE-MISSING", g, "exists/deprecate triple NOT in snapshot")
        if want_absent and is_present:
            add("TRIPLE-PRESENT", g, "add/mint+add/infer triple IS in snapshot")

    n_fail = sum(fails.values())
    return viol, {"gold": len(gold), "distinct_triples": len(want), "op_counts": counts,
                  "n_fail": n_fail, "fails": fails}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--snapshot-template", default="kg_snapshot_{year}-01-01.tsv")
    ap.add_argument("--eval-set-dir", required=True)
    ap.add_argument("--years", nargs="+", required=True)
    ap.add_argument("--max-examples", type=int, default=10, help="max example violations to print per kind/year")
    ap.add_argument("--warn-only", action="store_true", help="report but do NOT hard-fail (exit 0)")
    args = ap.parse_args()

    total_viol = 0
    summary = []
    for year in args.years:
        log(f"\n===================== YEAR {year} =====================")
        snap = os.path.join(args.snapshot_dir, args.snapshot_template.format(year=year))
        if not os.path.isfile(snap):
            raise SystemExit(f"FATAL: snapshot missing: {snap}")
        viol, stats = audit_year(snap, args.eval_set_dir, year, args.max_examples)
        summary.append((year, stats))
        if stats["n_fail"]:
            rate = 100 * stats["n_fail"] / stats["gold"] if stats["gold"] else 0
            log(f"    !!! {stats['n_fail']:,} violation(s) ({rate:.2f}% of {stats['gold']:,} gold) "
                f"by kind: {dict(stats['fails'])}; showing up to {args.max_examples}/kind:")
            for v in viol:
                log(f"      {v}")
        else:
            log("    OK: no inconsistencies")
        total_viol += stats["n_fail"]

    log("\n\n===================== AUDIT SUMMARY =====================")
    log(f"{'year':6} {'gold':>8} {'distinct':>9} {'violations':>11} {'fail%':>7}   op-counts")
    for year, stats in summary:
        opc = " ".join(f"{lab}={stats['op_counts'][lab]['n']}" for lab in OP_LABEL.values())
        rate = 100 * stats["n_fail"] / stats["gold"] if stats["gold"] else 0
        log(f"{year:6} {stats['gold']:>8,} {stats['distinct_triples']:>9,} {stats['n_fail']:>11,} {rate:>6.2f}%   {opc}")

    if total_viol:
        msg = f"\n{total_viol} violation example(s) across years -- eval-set/snapshot INCONSISTENT, investigate."
        if args.warn_only:
            log("WARN-ONLY:" + msg)
        else:
            raise SystemExit("FATAL:" + msg)
    else:
        log("\nALL YEARS CONSISTENT: entity/relation/triple presence matches the snapshots.")


if __name__ == "__main__":
    main()
