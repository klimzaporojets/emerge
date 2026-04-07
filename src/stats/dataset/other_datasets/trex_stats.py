#!/usr/bin/env python3
"""
trex_nif_stats_stream.py

Stream T-REx NIF (.ttl) inside TREx_NIF.zip and compute:
- total_instances: number of .ttl files scanned
- total_triples: number of extracted (subject, relation, object) facts (counts duplicates)
- num_relation_types: number of distinct relation/property IDs (Pxxx)
- num_distinct_entities: number of distinct entity IDs (Qxxx)

NOTE:
This is regex-based and does not require an RDF parser.
It assumes relations are Wikidata properties appearing as /prop/direct/P123 etc.
and entities appear as https?://www.wikidata.org/entity/Q123.

Also prints each .ttl filename as it is processed.

Usage:
  python trex_nif_stats_stream.py --zip /path/to/trex_nif.zip
"""

import argparse
import re
import zipfile


WIKIDATA_ENTITY_URI_RE = re.compile(r"https?://www\.wikidata\.org/entity/(Q[1-9]\d*)\b")

# --- ADDED: match Wikidata property IDs in common T-REx NIF predicate forms ---
# Examples:
#   http://www.wikidata.org/prop/direct/P31
#   https://www.wikidata.org/prop/direct/P17
#   http://www.wikidata.org/prop/P31  (less common, but harmless)
WIKIDATA_PROP_RE = re.compile(r"https?://www\.wikidata\.org/(?:prop/direct|prop)/(P[1-9]\d*)\b")

# --- ADDED: count TTL triples (RDF statements). This is an approximation:
# counts occurrences of " ." ending a statement in TTL.
# (Not perfect with multiline and literals, but usually good enough for rough triple count.)
TTL_STMT_END_RE = re.compile(r"\s\.\s*")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zip",
        required=False,
        help="Path to trex_nif.zip",
        default="/mnt/data/projects/msca-kgs/trex/TREx_NIF.zip",
    )
    ap.add_argument("--chunk_mb", type=int, default=4, help="Read size per chunk (MB)")
    args = ap.parse_args()

    # --- CHANGED: rename uniq -> entities; add rel_types; add triple counter ---
    entities = set()
    rel_types = set()
    total_triples = 0  # counts duplicates
    total_instances = 0  # number of .ttl files
    # --- END CHANGED ---

    n_bytes = 0
    chunk_size = args.chunk_mb * 1024 * 1024

    with zipfile.ZipFile(args.zip, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir():
                continue
            if not name.endswith(".ttl"):
                continue

            total_instances += 1
            print(
                f"reading: {name} , uniq_entities={len(entities)} , uniq_rel_types={len(rel_types)} , triples={total_triples}",
                flush=True,
            )

            with zf.open(info, "r") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    n_bytes += len(chunk)
                    text = chunk.decode("utf-8", errors="ignore")

                    # --- ADDED: entities ---
                    for qid in WIKIDATA_ENTITY_URI_RE.findall(text):
                        entities.add(qid)

                    # --- ADDED: relation types ---
                    for pid in WIKIDATA_PROP_RE.findall(text):
                        rel_types.add(pid)

                    # --- ADDED: triple count (approx: count statement terminators) ---
                    total_triples += len(TTL_STMT_END_RE.findall(text))

    print(f"\nzip: {args.zip}")
    print(f"ttl_files_scanned: {total_instances}")          # total instances
    print(f"scanned_gb: {n_bytes/1e9:.2f}")
    print(f"total_instances: {total_instances}")            # same as above, explicit
    print(f"total_triples: {total_triples}")                # approximate RDF statement count
    print(f"num_relation_types: {len(rel_types)}")          # distinct P-ids seen
    print(f"num_distinct_entities: {len(entities)}")        # distinct Q-ids seen
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
