"""Build a dense all-mpnet index over a KG snapshot's entity + relation vocabulary.

This is "retriever #2" (the all-mpnet alternative to reusing ReLiK's own index). It
implements the CORRECTED, GLOBAL design: we embed EVERY distinct entity label and
EVERY distinct relation label of a snapshot ONCE, so RETRIEVE can find candidates
index-wide by passage nearest-neighbor. NO graph traversal, no anchor neighborhoods.

Big one-time job (a snapshot has millions of distinct entities) -> run on GPU
(`--device cuda`); the code is device-configurable. For local dev, build a
small index from a `*_top_N.tsv` snapshot with `--device cpu`.

Outputs, per index, under <out_dir>/<kind>/ (kind = entities | relations):
  ids.txt     one QID/PID per line   (row order == embedding row)
  labels.txt  one label per line     (parallel to ids)
  emb.npy     float32 [N, dim], L2-normalized  (cosine == dot product)
  meta.json   {encoder, kind, n, dim, snapshot}

Query side = `kg_context.retriever.MpnetIndexRetriever`.

CLI:
    PYTHONPATH=src python -m kg_context.index_builder \
        --snapshot data/kg_snapshots/kg_snapshot_2019-01-01.tsv \
        --out-dir data/kg_context_index/mpnet/2019 --kind both --device cuda
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def extract_vocab(
    snapshot_path: str, kind: str, top_n_entities: int = -1, seed: int = 0
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Stream the snapshot -> distinct entities {qid:label} and relations {pid:label}.

    Entities = union of heads and tails (any node is a linkable candidate). 6-col gives
    labels; 3-col leaves label=None.

    `top_n_entities`: -1 = ALL entities (full index, the real build). N > 0 = a UNIFORM
    RANDOM sample of N distinct entities (NOT the first N -- stream order is alphabetical
    by QID, which would be a biased head). The returned dict IS the sampled set, so the
    downstream ids/labels/emb arrays stay aligned. Relations (only a few thousand) are
    always kept whole. `seed` makes the sample reproducible.
    """
    entities: dict[str, str | None] = {}
    relations: dict[str, str | None] = {}
    want_ent = kind in ("entities", "both")
    want_rel = kind in ("relations", "both")
    with open(snapshot_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            p = line.split("\t")
            if len(p) < 3:
                continue
            h, r, t = p[0], p[1], p[2]
            hl, rl, tl = (p[3], p[4], p[5]) if len(p) >= 6 else (None, None, None)
            if want_ent:
                entities.setdefault(h, hl)
                entities.setdefault(t, tl)
            if want_rel:
                relations.setdefault(r, rl)
    if want_ent and top_n_entities and top_n_entities > 0 and len(entities) > top_n_entities:
        import random
        keys = random.Random(seed).sample(list(entities.keys()), top_n_entities)
        entities = {k: entities[k] for k in keys}
    return entities, relations


def build_one(
    items: dict[str, str | None],
    encoder_name: str,
    out_dir: str,
    kind: str,
    device: str,
    batch_size: int,
    snapshot: str,
) -> None:
    """Embed the labels (fallback to id when no label) and save the index."""
    from sentence_transformers import SentenceTransformer

    os.makedirs(out_dir, exist_ok=True)
    ids = list(items.keys())
    labels = [items[i] if items[i] else i for i in ids]

    from tqdm import tqdm

    model = SentenceTransformer(encoder_name, device=device)
    dim = model.get_sentence_embedding_dimension()

    # memmap so we never hold the whole [N, dim] matrix in RAM (N can be millions)
    emb = np.lib.format.open_memmap(
        os.path.join(out_dir, "emb.npy"), mode="w+", dtype="float32", shape=(len(ids), dim)
    )
    for s in tqdm(range(0, len(ids), batch_size), desc=f"embed {kind}", unit="batch"):
        chunk = labels[s : s + batch_size]
        vecs = model.encode(
            chunk, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        emb[s : s + len(vecs)] = vecs
    emb.flush()

    with open(os.path.join(out_dir, "ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ids))
    with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"encoder": encoder_name, "kind": kind, "n": len(ids),
                   "dim": dim, "snapshot": snapshot}, f, indent=2)
    print(f"built {kind} index: {len(ids):,} items -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the all-mpnet entity/relation index.")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kind", choices=["entities", "relations", "both"], default="both")
    # content-affecting args are REQUIRED (no silent default that changes what/how-many gets embedded)
    ap.add_argument("--encoder", required=True, help="e.g. sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--top-n", type=int, required=True,
                    help="-1 = ALL entities; N>0 = a uniform RANDOM sample of N distinct entities (DEV index)")
    ap.add_argument("--seed", type=int, required=True, help="random-sample seed for --top-n")
    args = ap.parse_args()

    print(f"extracting vocab from {args.snapshot} (kind={args.kind}, top_n={args.top_n}, seed={args.seed}) ...")
    entities, relations = extract_vocab(args.snapshot, args.kind, top_n_entities=args.top_n, seed=args.seed)
    print(f"  distinct entities={len(entities):,} relations={len(relations):,}")
    if args.kind in ("entities", "both"):
        build_one(entities, args.encoder, os.path.join(args.out_dir, "entities"),
                  "entities", args.device, args.batch_size, args.snapshot)
    if args.kind in ("relations", "both"):
        build_one(relations, args.encoder, os.path.join(args.out_dir, "relations"),
                  "relations", args.device, args.batch_size, args.snapshot)


if __name__ == "__main__":
    main()
