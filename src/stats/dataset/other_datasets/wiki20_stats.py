#!/usr/bin/env python3
"""
wiki20_stats_from_txt_dir.py

Process all *.txt files in a directory. Each line is a JSON object like:
{"token":[...], "h":{"name":..,"id":..}, "t":{"name":..,"id":..}, "relation":"..."}

Compute dataset-wide stats (across all files):
- total_instances: number of JSON lines
- total_triples: same as total_instances (1 triple per instance; duplicates counted)
- num_relation_types: distinct relation strings
- num_distinct_entities: distinct entity IDs from h.id and t.id

Usage:
  python wiki20_stats_from_txt_dir.py --dir /path/to/wiki20
  python wiki20_stats_from_txt_dir.py --dir /path/to/wiki20 --entity_field name
"""

import argparse
import glob
import json
import os
from typing import Set


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",
                    default='/mnt/data/projects/msca-kgs/wiki20/wiki20m',
                    required=False,
                    help="Directory containing *.txt files")
    ap.add_argument(
        "--entity_field",
        default="id",
        choices=["id", "name"],
        help="Count distinct entities by this field in h/t (default: id)",
    )
    ap.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern within --dir (default: *.txt)",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files matched: {os.path.join(args.dir, args.pattern)}")

    total_instances = 0
    total_triples = 0  # 1 triple per line
    relation_types: Set[str] = set()
    entities: Set[str] = set()

    for fp in files:
        print(f"reading: {fp}", flush=True)
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)

                total_instances += 1
                total_triples += 1  # one (h, relation, t) per instance

                rel = ex.get("relation")
                if isinstance(rel, str) and rel:
                    relation_types.add(rel)

                h = ex.get("h")
                t = ex.get("t")
                if isinstance(h, dict):
                    v = h.get(args.entity_field)
                    if isinstance(v, str) and v:
                        entities.add(v)
                if isinstance(t, dict):
                    v = t.get(args.entity_field)
                    if isinstance(v, str) and v:
                        entities.add(v)

    print("\n== ALL FILES ==")
    print(f"total_instances: {total_instances}")
    print(f"total_triples: {total_triples}")
    print(f"num_relation_types: {len(relation_types)}")
    print(f"num_distinct_entities: {len(entities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
