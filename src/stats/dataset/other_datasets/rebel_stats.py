#!/usr/bin/env python3
"""
rebel_stats_rawzip.py

For Babelscape/rebel-dataset ZIP where JSONL lines have keys:
['docid', 'entities', 'text', 'title', 'triples', 'uri']

Computes in ONE pass (across en_train/en_val/en_test):
- total_instances: number of JSONL rows
- total_triples: total number of items in 'triples' across all rows (duplicates counted)
- num_relation_types: distinct predicate surfaceforms from triples
- num_distinct_entities: distinct entity surfaceforms from entities list

Requires:
  pip install -U huggingface_hub
"""

import argparse
import json
import zipfile
from typing import Set

from huggingface_hub import hf_hub_download


REPO_ID = "Babelscape/rebel-dataset"
ZIP_NAME = "rebel_dataset.zip"
SPLIT_FILES = ["en_train.jsonl", "en_val.jsonl", "en_test.jsonl"]


def main() -> int:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    zip_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=ZIP_NAME)

    total_instances = 0
    total_triples = 0
    relation_types: Set[str] = set()
    entities: Set[str] = set()

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for member in SPLIT_FILES:
            if member not in names:
                raise SystemExit(f"Missing {member} inside {ZIP_NAME}")

            print(f"reading: {member}", flush=True)
            with zf.open(member, "r") as f:
                for raw in f:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    ex = json.loads(line)
                    total_instances += 1

                    # --- entities (unique) ---
                    ents = ex.get("entities", [])
                    if isinstance(ents, list):
                        for e in ents:
                            if not isinstance(e, dict):
                                continue
                            sf = e.get("surfaceform")
                            if isinstance(sf, str) and sf:
                                entities.add(sf)

                    # --- triples + relation types ---
                    triples = ex.get("triples", [])
                    if isinstance(triples, list):
                        total_triples += len(triples)
                        for t in triples:
                            if not isinstance(t, dict):
                                continue
                            pred = t.get("predicate", {})
                            if isinstance(pred, dict):
                                p = pred.get("surfaceform")
                                if isinstance(p, str) and p:
                                    relation_types.add(p)

    print("\n== ALL SPLITS ==")
    print(f"total_instances: {total_instances}")
    print(f"total_triples: {total_triples}")
    print(f"num_relation_types: {len(relation_types)}")
    print(f"num_distinct_entities: {len(entities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
