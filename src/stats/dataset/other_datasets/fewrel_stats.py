#!/usr/bin/env python3
"""
fewrel_stats_noscripts.py

FewRel stats WITHOUT HuggingFace dataset scripts (works with new `datasets`):
- total_instances
- total_triples (counts duplicates; 1 per instance if relation != "")
- num_relation_types
- num_distinct_entities

It downloads the raw JSON files from the URLs used by the original dataset script.
"""

import argparse
import requests
from typing import Any, Dict, List, Set, Tuple


DATA_URL = "https://raw.githubusercontent.com/thunlp/FewRel/master/data/"
URLS = {
    "train_wiki": DATA_URL + "train_wiki.json",
    "val_nyt": DATA_URL + "val_nyt.json",
    "val_pubmed": DATA_URL + "val_pubmed.json",
    "val_semeval": DATA_URL + "val_semeval.json",
    "val_wiki": DATA_URL + "val_wiki.json",
    "pubmed_unsupervised": DATA_URL + "pubmed_unsupervised.json",
    # pid2name exists but not needed for these stats
}


def fetch_json(url: str, timeout_s: int = 60) -> Any:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def iter_instances(blob: Any, split_name: str):
    """
    Yield (relation, head_text, tail_text) for each instance in a FewRel split file.

    Format A (most splits): dict {relation: [instances...], ...}
      each instance has keys: tokens, h, t
      h = [text, type, indices], t = [text, type, indices]

    Format B (pubmed_unsupervised): list [instances...]
      relation is "" in HF script; we set relation=""
    """
    if isinstance(blob, dict):
        for rel, items in blob.items():
            if not isinstance(items, list):
                continue
            for it in items:
                # it["h"][0] and it["t"][0] are entity surface strings per script
                h = it.get("h", ["", "", []])
                t = it.get("t", ["", "", []])
                head_text = h[0] if isinstance(h, list) and len(h) > 0 else ""
                tail_text = t[0] if isinstance(t, list) and len(t) > 0 else ""
                yield rel, str(head_text), str(tail_text)
    elif isinstance(blob, list):
        for it in blob:
            h = it.get("h", ["", "", []])
            t = it.get("t", ["", "", []])
            head_text = h[0] if isinstance(h, list) and len(h) > 0 else ""
            tail_text = t[0] if isinstance(t, list) and len(t) > 0 else ""
            yield "", str(head_text), str(tail_text)
    else:
        return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--include_empty_relation",
        action="store_true",
        help="If set, include empty relation '' as a relation type (default: excluded)",
    )
    ap.add_argument(
        "--no_per_split",
        action="store_true",
        help="If set, only print ALL SPLITS totals",
    )
    args = ap.parse_args()

    dataset_total_instances = 0
    dataset_total_triples = 0
    dataset_relation_types: Set[str] = set()
    dataset_entities: Set[str] = set()

    for split_name, url in URLS.items():
        blob = fetch_json(url)

        split_instances = 0
        split_triples = 0
        split_relation_types: Set[str] = set()
        split_entities: Set[str] = set()

        for rel, head_text, tail_text in iter_instances(blob, split_name):
            split_instances += 1

            if head_text:
                split_entities.add(head_text)
            if tail_text:
                split_entities.add(tail_text)

            if rel != "" or args.include_empty_relation:
                split_relation_types.add(rel)

            if rel != "":
                split_triples += 1

        # accumulate dataset-wide
        dataset_total_instances += split_instances
        dataset_total_triples += split_triples
        dataset_relation_types |= split_relation_types
        dataset_entities |= split_entities

        if not args.no_per_split:
            print(f"\n== {split_name} ==")
            print(f"total_instances: {split_instances}")
            print(f"total_triples: {split_triples}")
            print(f"num_relation_types: {len(split_relation_types)}")
            print(f"num_distinct_entities: {len(split_entities)}")

    print(f"\n== ALL SPLITS ==")
    print(f"total_instances: {dataset_total_instances}")
    print(f"total_triples: {dataset_total_triples}")
    print(f"num_relation_types: {len(dataset_relation_types)}")
    print(f"num_distinct_entities: {len(dataset_entities)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
