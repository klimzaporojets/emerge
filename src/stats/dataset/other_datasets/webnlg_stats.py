#!/usr/bin/env python3
"""
webnlg_stats_noscripts.py

Load GEM WebNLG (en/ru) WITHOUT HuggingFace dataset scripts (works with newer `datasets`)
by downloading the official JSON files used by the original loader, then compute stats.

Usage:
  python webnlg_stats_noscripts.py --lang en
  python webnlg_stats_noscripts.py --lang ru
"""

import argparse
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import requests


URLS = {
    "en": {
        "train": "https://storage.googleapis.com/huggingface-nlp/datasets/gem/gem_web_nlg/webnlg_en_train.json",
        "validation": "https://storage.googleapis.com/huggingface-nlp/datasets/gem/gem_web_nlg/webnlg_en_val.json",
        "test": "https://storage.googleapis.com/huggingface-nlp/datasets/gem/gem_web_nlg/webnlg_en_test.json",
    },
    "ru": {
        "train": "https://storage.googleapis.com/huggingface-nlp/datasets/gem/gem_web_nlg/webnlg_ru_train.json",
        "validation": "https://storage.googleapis.com/huggingface-nlp/datasets/gem/gem_web_nlg/webnlg_ru_val.json",
        "test": "https://storage.googleapis.com/huggingface-nlp/datasets/gem/gem_web_nlg/webnlg_ru_test.json",
    },
}

TRIPLE_RE = re.compile(r"^\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*$")


def parse_predicate(triple_str: str) -> Optional[str]:
    """
    GEM WebNLG 'input' triples are typically strings: 'subj | property | obj'
    Returns the property/predicate if parseable, else None.
    """
    m = TRIPLE_RE.match(triple_str)
    if not m:
        return None
    return m.group(2)


def fetch_json(url: str, timeout_s: int = 60) -> Any:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def load_webnlg(lang: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Reproduces the core behavior of the official GEM WebNLG dataset script:

    For train:
      - expands each original record into multiple rows (one per target text)
      - references is [] (as in their script)

    For validation/test:
      - keeps one row per original record
      - target is the first reference (or "" if none)
      - references is the full target list
    """
    if lang not in URLS:
        raise ValueError(f"lang must be one of {list(URLS.keys())}, got: {lang}")

    out: Dict[str, List[Dict[str, Any]]] = {}

    for split, url in URLS[lang].items():
        blob = fetch_json(url)
        examples = blob["values"]

        rows: List[Dict[str, Any]] = []
        id_ = -1

        for ex in examples:
            inp = ex["input"]          # list[str]
            targets = ex["target"]     # list[str]
            category = ex["category"]
            webnlg_id = ex["webnlg-id"]

            if split == "train":
                for t in targets:
                    id_ += 1
                    gid = f"web_nlg_{lang}-{split}-{id_}"
                    rows.append({
                        "gem_id": gid,
                        "gem_parent_id": gid,
                        "input": inp,
                        "target": t,
                        "references": [],
                        "category": category,
                        "webnlg_id": webnlg_id,
                    })
            else:
                id_ += 1
                gid = f"web_nlg_{lang}-{split}-{id_}"
                rows.append({
                    "gem_id": gid,
                    "gem_parent_id": gid,
                    "input": inp,
                    "target": targets[0] if targets else "",
                    "references": targets,
                    "category": category,
                    "webnlg_id": webnlg_id,
                })

        out[split] = rows

    return out

# --- MINIMAL MODS ONLY (add relation types + distinct entities; keep triples) ---
# Changes:
# 1) Add a helper to parse full triple into (subj, pred, obj)
# 2) Track sets of relation types + entities in compute_stats
# 3) Return those sets’ sizes and accumulate dataset-wide in main

# --- ADDED: parse full triple (not just predicate) ---
def parse_triple(triple_str: str) -> Optional[Tuple[str, str, str]]:
    """
    GEM WebNLG 'input' triples are typically strings: 'subj | property | obj'
    Returns (subj, pred, obj) if parseable, else None.
    """
    m = TRIPLE_RE.match(triple_str)
    if not m:
        return None
    return (m.group(1), m.group(2), m.group(3))
# --- END ADDED ---


# --- CHANGED: add return values for relation types + entity types ---
def compute_stats(
    rows: List[Dict[str, Any]],
    split_name: str
) -> Tuple[Dict[str, float], Counter, int, int, int]:
    pred = Counter()

    n_examples = len(rows)
    n_triples_list: List[int] = []
    n_refs_list: List[int] = []
    ref_lens: List[int] = []

    total_triples = 0  # total number of triples in this split (counts duplicates)

    rel_types = set()  # <-- ADDED: distinct relation/property types in this split
    entities = set()   # <-- ADDED: distinct entities (subjects + objects) in this split

    for r in rows:
        triples = r.get("input") or []
        n_triples_list.append(len(triples))

        total_triples += len(triples)

        for t in triples:
            if not isinstance(t, str):
                continue

            # --- ADDED: extract (s, p, o) and update sets ---
            spo = parse_triple(t)
            if spo is not None:
                s, p, o = spo
                rel_types.add(p)
                entities.add(s)
                entities.add(o)
            # --- END ADDED ---

            p = parse_predicate(t)
            if p is not None:
                pred[p] += 1

        if split_name == "train":
            texts = [r.get("target", "")]
            n_refs_list.append(1)
        else:
            refs = r.get("references") or []
            n_refs_list.append(len(refs))
            texts = refs if refs else [r.get("target", "")]

        for txt in texts:
            txt = txt or ""
            ref_lens.append(len(str(txt).split()))

    def mean(xs: List[int]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    summary = {
        "num_examples": float(n_examples),
        "avg_triples_per_ex": mean(n_triples_list),
        "triples_min": float(min(n_triples_list)) if n_triples_list else 0.0,
        "triples_max": float(max(n_triples_list)) if n_triples_list else 0.0,
        "avg_refs_per_ex": mean(n_refs_list),
        "avg_ref_len_words": mean(ref_lens),
        "num_unique_predicates": float(len(pred)),
        "total_triples": float(total_triples),

        "num_relation_types": float(len(rel_types)),   # <-- ADDED
        "num_distinct_entities": float(len(entities)), # <-- ADDED
    }

    # --- CHANGED: return sizes so main can accumulate dataset-wide totals ---
    return summary, pred, total_triples, len(rel_types), len(entities)
# --- END CHANGED ---


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=["en"], help="Dataset language")
    ap.add_argument("--topk", type=int, default=10, help="Top-K predicates to print")
    args = ap.parse_args()
    dataset_total_instances = 0  # <-- ADDED: total number of rows across all splits

    ds = load_webnlg(args.lang)

    print(f"Language: {args.lang}")
    print(f"Splits: {list(ds.keys())}")

    dataset_total_triples = 0

    # --- ADDED: dataset-wide sets (NOT sum) for distinct counts across splits ---
    dataset_rel_types = set()
    dataset_entities = set()
    # --- END ADDED ---

    for split_name, rows in ds.items():
        dataset_total_instances += len(rows)
        # --- CHANGED: capture rel/entity counts (and also update dataset-wide sets below) ---
        summary, pred, total_triples, _, _ = compute_stats(rows, split_name)
        # --- END CHANGED ---

        dataset_total_triples += total_triples

        # --- ADDED: update dataset-wide sets by scanning split rows once (minimal extra logic) ---
        # (We re-scan triples here to avoid returning huge sets from compute_stats.)
        for r in rows:
            for t in (r.get("input") or []):
                if not isinstance(t, str):
                    continue
                spo = parse_triple(t)
                if spo is None:
                    continue
                s, p, o = spo
                dataset_rel_types.add(p)
                dataset_entities.add(s)
                dataset_entities.add(o)
        # --- END ADDED ---

        print(f"\n== {split_name} ==")
        print(f"total_triples: {total_triples}")
        print(f"num_relation_types: {int(summary['num_relation_types'])}")     # <-- ADDED
        print(f"num_distinct_entities: {int(summary['num_distinct_entities'])}")  # <-- ADDED

        for k in [
            "num_examples",
            "avg_triples_per_ex",
            "triples_min",
            "triples_max",
            "avg_refs_per_ex",
            "avg_ref_len_words",
            "num_unique_predicates",
        ]:
            v = summary[k]
            if k in {"num_examples", "triples_min", "triples_max", "num_unique_predicates"}:
                print(f"{k}: {int(v)}")
            else:
                print(f"{k}: {v:.4f}")

        print(f"top predicates (top {args.topk}):")
        for p, c in pred.most_common(args.topk):
            print(f"  {p}\t{c}")

    print(f"\n== ALL SPLITS ==")
    print(f"total_triples: {dataset_total_triples}")
    print(f"num_relation_types: {len(dataset_rel_types)}")      # <-- ADDED
    print(f"num_distinct_entities: {len(dataset_entities)}")    # <-- ADDED
    print(f"dataset_total_instances: {dataset_total_instances}")    # <-- ADDED

    return 0




if __name__ == "__main__":
    main()
