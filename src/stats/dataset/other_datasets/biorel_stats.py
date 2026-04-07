#!/usr/bin/env python3
"""
biorel_stats_parquet.py

Compute BioRel stats from HF parquet files (no dataset scripts):

Stats (dataset-wide across train/validation/test):
- total_instances
- total_triples            (default: count only relation != "NA" as a triple)
- num_relation_types       (default: excludes "NA")
- num_distinct_entities    (default: distinct entity IDs from h.id and t.id)

Requires:
  pip install -U huggingface_hub pyarrow

Usage:
  python biorel_stats_parquet.py
  python biorel_stats_parquet.py --include_na_relation
  python biorel_stats_parquet.py --count_na_as_triple
  python biorel_stats_parquet.py --entity_field name
"""

import argparse
from typing import Set

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


REPO_ID = "DFKI-SLT/BioRel"
FILENAMES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}


def download_split(split: str) -> str:
    return hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=FILENAMES[split])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--include_na_relation",
        action="store_true",
        help='Include "NA" in relation-type count (default: excluded).',
    )
    ap.add_argument(
        "--count_na_as_triple",
        action="store_true",
        help='Count rows with relation=="NA" as triples too (default: not counted).',
    )
    ap.add_argument(
        "--entity_field",
        default="id",
        choices=["id", "name"],
        help="Count distinct entities by this field inside h/t dict (default: id).",
    )
    args = ap.parse_args()

    total_instances = 0
    total_triples = 0
    relation_types: Set[str] = set()
    entities: Set[str] = set()

    for split in ["train", "validation", "test"]:
        path = download_split(split)
        pf = pq.ParquetFile(path)

        split_instances = 0
        split_triples = 0
        split_rel_types: Set[str] = set()
        split_entities: Set[str] = set()

        # Stream through parquet in record batches
        for batch in pf.iter_batches(batch_size=65536, columns=["relation", "h", "t"]):
            rel_col = batch.column(batch.schema.get_field_index("relation")).to_pylist()
            h_col = batch.column(batch.schema.get_field_index("h")).to_pylist()
            t_col = batch.column(batch.schema.get_field_index("t")).to_pylist()

            split_instances += len(rel_col)

            for rel, h, t in zip(rel_col, h_col, t_col):
                # relation types
                if rel is not None:
                    if rel != "NA" or args.include_na_relation:
                        split_rel_types.add(rel)

                # triples: 1 per row if relation != "NA" (unless overridden)
                if args.count_na_as_triple:
                    split_triples += 1
                else:
                    if rel is not None and rel != "NA":
                        split_triples += 1

                # entities from h/t dicts
                if isinstance(h, dict):
                    v = h.get(args.entity_field)
                    if isinstance(v, str) and v:
                        split_entities.add(v)
                if isinstance(t, dict):
                    v = t.get(args.entity_field)
                    if isinstance(v, str) and v:
                        split_entities.add(v)

        # accumulate dataset-wide
        total_instances += split_instances
        total_triples += split_triples
        relation_types |= split_rel_types
        entities |= split_entities

        print(f"\n== {split} ==")
        print(f"instances: {split_instances}")
        print(f"triples: {split_triples}")
        print(f"relation_types: {len(split_rel_types)}")
        print(f"distinct_entities: {len(split_entities)}")

    print("\n== ALL SPLITS ==")
    print(f"total_instances: {total_instances}")
    print(f"total_triples: {total_triples}")
    print(f"num_relation_types: {len(relation_types)}")
    print(f"num_distinct_entities: {len(entities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
