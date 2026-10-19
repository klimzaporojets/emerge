#!/usr/bin/env python3
"""Pipeline step s08b: build a ReLiK native entity index (`embeddings.pt`) from
its dictionary (`documents.jsonl`) for one KG snapshot.

Runs AFTER `s08_extract_relik_index` (which produces the per-snapshot dictionary):
s08 extracts the dictionary, s08b encodes it into the searchable index.

WHY THIS EXISTS
---------------
The KG-aware (KGA) baseline's AUGMENT step retrieves candidate entities with
ReLiK over a per-snapshot entity index that has two parts:

    documents.jsonl   the entity "dictionary": {text, id, metadata:{wikidata, definition}}
    embeddings.pt     float32 [N, 384] -- the dictionary encoded by the ReLiK
                      entity encoder (e5-small-v2 matryoshka, 384-dim)

`documents.jsonl` is text and compresses ~2.7x; `embeddings.pt` is dense float32
and barely compresses. So only the gzipped dictionaries are distributed, and this
script regenerates the embeddings locally on a GPU (a few hours per snapshot for
the full ~6M-entity dictionary; minutes on a small one).

Encoding is deterministic, so the rebuilt `embeddings.pt` reproduces the released
index exactly (validated at cosine 1.0000 row-for-row).

This is a thin wrapper around ReLiK's own `create-index` routine
(`relik.cli.retriever.create_index`) -- the exact routine the released indexes
were built with -- imported directly so it does not need ReLiK's `relik` console
entry point (which pulls in gradio).

USAGE
-----
    conda activate relik                       # needs the `relik` package (1.0.x)
    python scripts/kg_context/s08b_build_relik_index.py \
        --documents data/indices/relik_entity_index/2019-01-01/documents.jsonl.gz \
        --out-dir   data/indices/relik_entity_index/2019-01-01 \
        --device    cuda

`--documents` may be a plain `.jsonl` or a gzipped `.jsonl.gz` (auto-detected).
After this writes `embeddings.pt` (+ `documents.jsonl` + `config.yaml`) into
--out-dir, AUGMENT can load the snapshot via the usual config.

Validate against an existing embeddings.pt on the first --validate-n rows
(cosine ~1.0 => exact rebuild):
    ... --validate-against data/indices/relik_entity_index/2019-01-01/embeddings.pt --validate-n 256

The optional AIDA entity index uses a different encoder; pass
`--encoder sapienzanlp/relik-entity-linking-large` for that ablation.
"""
import argparse
import gzip
import os
import shutil
import tempfile

# Entity encoder for the NATIVE index (relik-cie-large's span encoder). This is
# the encoder the released per-snapshot entity indexes were built with.
DEFAULT_ENCODER = "relik-ie/encoder-e5-small-v2-wikipedia-matryoshka"


def _materialize_documents(path: str, workdir: str) -> str:
    """Return a path to a plain .jsonl (decompressing a .gz into workdir if needed)."""
    if path.endswith(".gz"):
        out = os.path.join(workdir, "documents.jsonl")
        print(f"[build] decompressing {path} -> {out}", flush=True)
        with gzip.open(path, "rb") as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return out
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--documents", required=True,
                    help="Path to the snapshot's documents.jsonl (or documents.jsonl.gz).")
    ap.add_argument("--out-dir", required=True,
                    help="Output index dir (gets documents.jsonl + embeddings.pt + config.yaml).")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER,
                    help=f"ReLiK entity encoder (default: {DEFAULT_ENCODER}).")
    ap.add_argument("--device", default="cuda", help="cuda | cpu (default: cuda).")
    ap.add_argument("--batch-size", type=int, default=512, help="Encoding batch size (default: 512).")
    ap.add_argument("--passage-max-length", type=int, default=64,
                    help="Max passage length in tokens (default: 64, ReLiK's create-index default).")
    ap.add_argument("--precision", default="fp32",
                    help="Encoding precision: fp32 reproduces the released index exactly (default: fp32).")
    ap.add_argument("--validate-against", default=None,
                    help="Optional existing embeddings.pt to sanity-check the first --validate-n rows against.")
    ap.add_argument("--validate-n", type=int, default=256, help="Rows to compare in --validate-against mode.")
    args = ap.parse_args()

    # Import the routine directly (NOT via the `relik` console script, which imports gradio).
    from relik.cli.retriever import create_index

    workdir = tempfile.mkdtemp(prefix="s08b_build_relik_index_")
    try:
        docs = _materialize_documents(args.documents, workdir)
        os.makedirs(args.out_dir, exist_ok=True)

        print(f"[build] encoding {docs} with {args.encoder} on {args.device} "
              f"(batch_size={args.batch_size}, max_length={args.passage_max_length}, precision={args.precision}) ...",
              flush=True)
        create_index(
            question_encoder_name_or_path=args.encoder,
            document_path=docs,
            output_folder=args.out_dir,
            document_file_type="jsonl",
            batch_size=args.batch_size,
            passage_max_length=args.passage_max_length,
            device=args.device,
            precision=args.precision,
        )
        print(f"[build] wrote index to {args.out_dir} (embeddings.pt + documents.jsonl + config.yaml)", flush=True)

        if args.validate_against:
            import torch
            import torch.nn.functional as F
            emb = torch.load(os.path.join(args.out_dir, "embeddings.pt"), map_location="cpu", mmap=True)
            k = min(args.validate_n, emb.shape[0])
            got = emb[:k].float()
            ref = torch.load(args.validate_against, map_location="cpu", mmap=True)[:k].float()
            cos = (F.normalize(got, dim=1) * F.normalize(ref, dim=1)).sum(1)
            print(f"[build] validate first {k} rows: cosine min={cos.min():.4f} mean={cos.mean():.4f}")
            if cos.mean() <= 0.999:
                print("[build] WARNING: rebuilt embeddings do not match the reference index.")
                return 1
            print("[build] OK: rebuilt embeddings match the reference index.")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
