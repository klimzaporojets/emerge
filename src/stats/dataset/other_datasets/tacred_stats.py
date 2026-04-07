#!/usr/bin/env python3
"""
tacred_stats.py

Compute TACRED stats:
- total_instances (rows across splits)
- total_triples (counts duplicates; 1 per instance if relation != no_relation)
- num_relation_types (distinct relation labels)
- num_distinct_entities (distinct subj + obj strings from token spans)

Requires manual TACRED download (LDC2018T24):
  datasets.load_dataset("DFKI-SLT/tacred", data_dir="/path/to/unzipped_tacred")
"""

import argparse
from typing import Set, Tuple

from datasets import load_dataset


def span_text(tokens, start: int, end: int) -> str:
    # subj_end/obj_end are exclusive in the HF script
    # Join with spaces; good enough for "distinct entities" counting
    return " ".join(tokens[start:end]).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        required=True,
        help="Path to folder containing TACRED train.json/dev.json/test.json (unzipped LDC2018T24)",
    )
    ap.add_argument(
        "--config",
        default="original",
        choices=["original", "revisited", "re-tacred"],
        help="Which TACRED config to load",
    )
    ap.add_argument(
        "--include_no_relation",
        action="store_true",
        help="If set, count 'no_relation' as a relation type (default: excluded from relation-type count)",
    )
    args = ap.parse_args()

    ds = load_dataset("DFKI-SLT/tacred", name=args.config, data_dir=args.data_dir)

    total_instances = 0
    total_triples = 0  # 1 per instance if relation != no_relation (duplicates allowed)
    relation_types: Set[str] = set()
    entities: Set[str] = set()

    for split_name in ds.keys():
        split = ds[split_name]
        total_instances += len(split)

        for ex in split:
            rel = ex["relation"]  # string label
            tokens = ex["token"]

            subj = span_text(tokens, ex["subj_start"], ex["subj_end"])
            obj = span_text(tokens, ex["obj_start"], ex["obj_end"])

            if subj:
                entities.add(subj)
            if obj:
                entities.add(obj)

            # relation types
            if rel != "no_relation" or args.include_no_relation:
                relation_types.add(rel)

            # triples: count one labeled triple per instance if it's not no_relation
            if rel != "no_relation":
                total_triples += 1

    print(f"config: {args.config}")
    print(f"total_instances: {total_instances}")
    print(f"total_triples: {total_triples}")
    print(f"num_relation_types: {len(relation_types)}")
    print(f"num_distinct_entities: {len(entities)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
