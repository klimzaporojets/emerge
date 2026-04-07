#!/usr/bin/env python3
"""
text2kg_stats_from_split_dirs.py

Given a root directory, recursively find directories named:
  train, validation, test
and parse ALL *.jsonl files inside them (recursively).

Each JSONL line is an instance like:
{
  "id": "...",
  "sent": "...",
  "triples": [{"sub": "...", "rel": "...", "obj": "..."}, ...]
}

Stats (across ALL found split dirs):
- total_instances: number of JSONL lines
- total_triples: sum of len(triples) across all instances (duplicates counted)
- num_relation_types: distinct rel values
- num_distinct_entities: distinct entity strings from sub and obj

Usage:
  python text2kg_stats_from_split_dirs.py --root /path/to/text2kg_root
"""

import argparse
import json
import os
from typing import List, Set, Tuple


SPLIT_DIRS = {"train", "validation", "test"}


def find_split_dirs(root: str) -> List[str]:
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        # if current directory itself is a split dir
        base = os.path.basename(dirpath)
        if base in SPLIT_DIRS:
            out.append(dirpath)
            # don't stop; there might be nested split dirs too
    return sorted(set(out))


def iter_jsonl_files_under(d: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(d):
        for fn in filenames:
            if fn.endswith(".jsonl"):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=False,
                    default='/path/to/data
                    help="Root directory to search recursively")
    args = ap.parse_args()

    split_dirs = find_split_dirs(args.root)
    if not split_dirs:
        raise SystemExit(f"No split dirs named {sorted(SPLIT_DIRS)} found under: {args.root}")

    jsonl_files: List[str] = []
    for sd in split_dirs:
        jsonl_files.extend(iter_jsonl_files_under(sd))

    if not jsonl_files:
        raise SystemExit(f"Found split dirs but no .jsonl files under them. root={args.root}")

    total_instances = 0
    total_triples = 0
    relation_types: Set[str] = set()
    entities: Set[str] = set()

    for fp in jsonl_files:
        print(f"reading: {fp}", flush=True)
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                total_instances += 1

                triples = ex.get("triples", [])
                if not isinstance(triples, list):
                    continue

                total_triples += len(triples)

                for t in triples:
                    if not isinstance(t, dict):
                        continue
                    sub = t.get("sub")
                    rel = t.get("rel")
                    obj = t.get("obj")

                    if isinstance(rel, str) and rel:
                        relation_types.add(rel)
                    if isinstance(sub, str) and sub:
                        entities.add(sub)
                    if isinstance(obj, str) and obj:
                        entities.add(obj)

    print("\n== ALL SPLITS (train/validation/test) ==")
    print(f"split_dirs_found: {len(split_dirs)}")
    print(f"jsonl_files_found: {len(jsonl_files)}")
    print(f"total_instances: {total_instances}")
    print(f"total_triples: {total_triples}")
    print(f"num_relation_types: {len(relation_types)}")
    print(f"num_distinct_entities: {len(entities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
