#!/usr/bin/env python3
"""
docred_like_stats_stream.py

Streaming stats for huge JSON with format:
[
  { "vertexSet": [...], "labels": [...], ... },
  ...
]

Stats:
- total_instances: number of documents (top-level items)
- total_triples: total number of label entries across all docs (counts duplicates)
- num_relation_types: distinct labels[*].r
- num_distinct_entities: distinct entity names from vertexSet[*][*].name

Requires:
  pip install ijson

Usage:
  python docred_like_stats_stream.py --input /path/to/data.json
"""

import argparse
from typing import Set

import ijson


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    required=False,
                    default='/mnt/data/projects/msca-kgs/docred/train_distant.json',
                    help="Path to huge JSON file")
    ap.add_argument(
        "--entity_key",
        default="name",
        help="Key inside vertex mentions to count entities by (default: name)",
    )
    args = ap.parse_args()

    total_instances = 0
    total_triples = 0
    relation_types: Set[str] = set()
    entities: Set[str] = set()

    # Stream each top-level item of the JSON array:
    # ijson.items(f, "item") yields dicts for each element in the top-level list.
    with open(args.input, "rb") as f:
        for doc in ijson.items(f, "item"):
            total_instances += 1

            # entities: vertexSet is list of entities, each is list of mentions (dicts)
            vset = doc.get("vertexSet", [])
            if isinstance(vset, list):
                for ent in vset:
                    if not isinstance(ent, list):
                        continue
                    for mention in ent:
                        if not isinstance(mention, dict):
                            continue
                        val = mention.get(args.entity_key)
                        if isinstance(val, str) and val:
                            entities.add(val)

            # triples + relation types from labels
            labels = doc.get("labels", [])
            if isinstance(labels, list):
                total_triples += len(labels)
                for lab in labels:
                    if not isinstance(lab, dict):
                        continue
                    r = lab.get("r")
                    if isinstance(r, str) and r:
                        relation_types.add(r)

            if total_instances % 10000 == 0:
                print(
                    f"docs={total_instances} "
                    f"triples={total_triples} "
                    f"rel_types={len(relation_types)} "
                    f"entities={len(entities)}",
                    flush=True,
                )

    print(f"total_instances: {total_instances}")
    print(f"total_triples: {total_triples}")
    print(f"num_relation_types: {len(relation_types)}")
    print(f"num_distinct_entities: {len(entities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
