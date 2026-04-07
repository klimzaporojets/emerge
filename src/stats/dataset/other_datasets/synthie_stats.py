#!/usr/bin/env python3
"""
synthie_stats_noscripts.py

NO ARGS.

Computes the 4 stats for:
  - synthie_code
  - synthie_text
  - rebel
and prints:
  - per-config stats
  - SUM across the three configs:
      * total_instances (sum)
      * total_triples (sum)
      * num_relation_types (union)
      * num_distinct_entities (union)

Counts entities/relations by URI (canonical), like before.

Requires:
  pip install -U huggingface_hub jsonlines
"""

import gzip
from typing import Any, Dict, List, Set, Tuple

import jsonlines
from huggingface_hub import hf_hub_download


REPO_ID = "martinjosifoski/SynthIE"

CONFIGS = ["synthie_code", "synthie_text", "rebel"]

dataset_name2folder_name = {
    "synthie_text": "sdg_text_davinci_003",
    "synthie_text_pc": "sdg_text_davinci_003",
    "synthie_code": "sdg_code_davinci_002",
    "synthie_code_pc": "sdg_code_davinci_002",
    "rebel": "rebel",
    "rebel_pc": "rebel",
}

flag_ordered = {
    "synthie_text": False,
    "synthie_code": False,
    "rebel": False,
    "synthie_text_pc": True,
    "synthie_code_pc": True,
    "rebel_pc": False,
}

data_url_prefix = {
    "synthie_text": "",
    "synthie_code": "",
    "rebel": "",
    "synthie_text_pc": "processed",
    "synthie_code_pc": "processed",
    "rebel_pc": "processed",
}


def split_to_filename(split: str, ordered: bool) -> str:
    return f"{split}_ordered.jsonl.gz" if ordered else f"{split}.jsonl.gz"


def splits_for_config(cfg: str) -> List[str]:
    base = ["val", "test", "test_small"]
    if cfg != "synthie_text":  # synthie_text has no train
        return ["train"] + base
    return base


def hf_path(cfg: str, split: str) -> str:
    prefix = data_url_prefix[cfg]
    folder = dataset_name2folder_name[cfg]
    fname = split_to_filename(split, flag_ordered[cfg])
    return f"{prefix}/{folder}/{fname}" if prefix else f"{folder}/{fname}"


def get_str(x: Any) -> str:
    return x if isinstance(x, str) else ""


def compute_config(cfg: str) -> Tuple[int, int, Set[str], Set[str]]:
    total_instances = 0
    total_triples = 0
    entities: Set[str] = set()   # URIs
    rel_types: Set[str] = set()  # URIs

    for split in splits_for_config(cfg):
        path_in_repo = hf_path(cfg, split)
        local = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=path_in_repo)
        print(f"reading: {cfg} / {path_in_repo}", flush=True)

        with gzip.open(local, "rb") as fp:
            reader = jsonlines.Reader(fp)
            for sample in reader:
                total_instances += 1

                triplets = sample.get("triplets", [])
                if isinstance(triplets, list):
                    total_triples += len(triplets)

                ent_list = sample.get("entities", [])
                if isinstance(ent_list, list):
                    for e in ent_list:
                        if isinstance(e, dict):
                            v = get_str(e.get("uri"))
                            if v:
                                entities.add(v)

                rel_list = sample.get("relations", [])
                if isinstance(rel_list, list):
                    for r in rel_list:
                        if isinstance(r, dict):
                            v = get_str(r.get("uri"))
                            if v:
                                rel_types.add(v)

    return total_instances, total_triples, rel_types, entities


def main() -> int:
    # dataset-wide unions
    all_instances = 0
    all_triples = 0
    all_rel_types: Set[str] = set()
    all_entities: Set[str] = set()

    for cfg in CONFIGS:
        inst, trp, rels, ents = compute_config(cfg)

        print(f"\n== {cfg} ==")
        print(f"total_instances: {inst}")
        print(f"total_triples: {trp}")
        print(f"num_relation_types: {len(rels)}")
        print(f"num_distinct_entities: {len(ents)}")

        all_instances += inst
        all_triples += trp
        all_rel_types |= rels
        all_entities |= ents

    print("\n== SUM (synthie_code + synthie_text + rebel) ==")
    print(f"total_instances: {all_instances}")
    print(f"total_triples: {all_triples}")
    print(f"num_relation_types: {len(all_rel_types)}")
    print(f"num_distinct_entities: {len(all_entities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
