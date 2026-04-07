#!/usr/bin/env python3
"""
chemdisgene_stats_from_tsv_dir.py

ChemDisGene stats from *.tsv files in a directory.

CHANGE REQUEST:
- total_instances should be the number of DISTINCT PMIDs (first TSV field)

Other stats unchanged:
- total_triples: number of TSV rows (duplicates counted)
- num_relation_types
- num_distinct_entities
"""

import argparse
import glob
import os
from typing import Set


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default="/path/to/data",
        required=False,
        help="Directory containing *.tsv files",
    )
    ap.add_argument("--pattern", default="*.tsv", help="Glob pattern inside --dir (default: *.tsv)")
    ap.add_argument("--skip_header", action="store_true", help="Skip first line of each file (if header exists)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files matched: {os.path.join(args.dir, args.pattern)}")

    # --- CHANGED: track distinct PMIDs for instances ---
    pmids: Set[str] = set()
    # --- END CHANGED ---

    total_triples = 0  # 1 per row
    relation_types: Set[str] = set()
    entities: Set[str] = set()

    for fp in files:
        print(f"reading: {fp}", flush=True)
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            first = True
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if first and args.skip_header:
                    first = False
                    continue
                first = False

                parts = line.split("\t")
                if len(parts) < 4:
                    continue

                # --- CHANGED: PMID is instances key ---
                pmid = parts[0].strip()
                if pmid:
                    pmids.add(pmid)
                # --- END CHANGED ---

                # columns: pmid, relation, head_id, tail_id
                rel = parts[1].strip()
                h = parts[2].strip()
                t = parts[3].strip()

                # --- CHANGED: instances no longer increment per row ---
                total_triples += 1
                # --- END CHANGED ---

                if rel:
                    relation_types.add(rel)
                if h:
                    entities.add(h)
                if t:
                    entities.add(t)

    print("\n== ALL FILES ==")
    # --- CHANGED: instances = distinct PMIDs ---
    print(f"total_instances: {len(pmids)}")
    # --- END CHANGED ---
    print(f"total_triples: {total_triples}")
    print(f"num_relation_types: {len(relation_types)}")
    print(f"num_distinct_entities: {len(entities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
