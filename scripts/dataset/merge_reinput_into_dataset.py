#!/usr/bin/env python3
"""Merge corrected records from a reinput-tree concat file into a NEW
dataset tree, splicing by `hash_id`. Source tree stays read-only.

Companion to build_reinput_for_garbage.py. After a single-sbatch
re-query has produced corrected 405B-v1 assessments at e.g.
`…/_complete_dataset_reinput_garbage_2_concat/.../delta_2099-01-01.jsonl`,
this script:

  1. Reads that "corrections" file and indexes records by `hash_id`.
  2. Walks the source tree's `delta_*.jsonl` files in
     `snapshot_*-01-01/llm_assessed/`.
  3. For each source record:
       - If its `hash_id` is in the corrections index, writes the
         CORRECTED record (full object replacement) to the merged tree.
       - Else, writes the source line BYTE-FOR-BYTE unchanged.
  4. Output: a NEW tree at `--output-root` mirroring the source layout,
     where corrected records replace source records of the same hash.

The source tree is byte-for-byte unchanged. Hard-asserts on path
disjointness.

Whole-record replacement is correct because the reinput record contains
the source record's original fields (chunk, mentions, 8B assessments,
…) PLUS a freshly-queried `Meta-Llama-3.1-405B_prompt_v1` entry that
overwrites any garbage version. The pipeline's per-triple skip logic
preserves prior assessments while adding new ones, so no information
is lost in the splice.

Typical usage (read-only on source; ~5-10 min on a CPU node):

    python scripts/dataset/merge_reinput_into_dataset.py \\
        --source-root <your-output-tree>/llama405b_assessed \\
        --reinput-file <reinput-rerun-output>/llama405b_assessed/snapshot_2099-01-01/llm_assessed/delta_2099-01-01.jsonl \\
        --output-root <merged-tree>/llama405b_assessed \\
        --output-manifest output/merge_manifest_$(date +%Y%m%d_%H%M).json

Add `--dry-run` to preview without writing anything.
Add `--force` if `--output-root` already exists and is non-empty.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def is_subpath(child, parent):
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_paths_safe(source_root, output_root, reinput_file):
    if not source_root.is_dir():
        sys.exit(f"ERROR: --source-root does not exist or is not a directory: {source_root}")
    if not reinput_file.is_file():
        sys.exit(f"ERROR: --reinput-file does not exist: {reinput_file}")
    if source_root.resolve() == output_root.resolve():
        sys.exit(f"ERROR: --source-root and --output-root resolve to the same path: {source_root}")
    if is_subpath(output_root, source_root):
        sys.exit(f"ERROR: --output-root is inside --source-root. This is unsafe.\n"
                 f"  source = {source_root}\n  output = {output_root}")
    if is_subpath(source_root, output_root):
        sys.exit(f"ERROR: --source-root is inside --output-root. This is unsafe.\n"
                 f"  source = {source_root}\n  output = {output_root}")
    if is_subpath(reinput_file, source_root):
        sys.exit(f"ERROR: --reinput-file is inside --source-root. Move it outside.")


def build_corrections_index(reinput_file):
    """Stream the reinput file; build {hash_id -> raw_json_line}.

    Stores the original JSON line string (not the parsed object) so the
    splice can write the corrected record byte-for-byte from how the
    pipeline produced it (no risk of key-order drift from parse+dump).
    Asserts hash_id uniqueness within the reinput file.
    """
    index = {}
    duplicates = []
    with reinput_file.open() as fh:
        for line_idx, raw in enumerate(fh):
            line = raw.rstrip("\n")
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"ERROR: malformed JSON at line {line_idx} of {reinput_file}: {e}")
            hash_id = record.get("hash_id")
            if hash_id is None:
                sys.exit(f"ERROR: line {line_idx} of {reinput_file} has no hash_id")
            if hash_id in index:
                duplicates.append(hash_id)
                continue
            index[hash_id] = line
    if duplicates:
        sys.exit(f"ERROR: {len(duplicates)} duplicate hash_id(s) in --reinput-file.\n"
                 f"  Sample: {duplicates[:5]}\n"
                 f"  This violates the upstream concat's uniqueness invariant.")
    return index


def merge_delta_file(source_path, output_path, corrections_index,
                     substituted_hash_ids, multiply_used_hash_ids, dry_run):
    """Stream source_path; for each line, substitute via corrections_index
    by hash_id or pass through unchanged. Writes to output_path via
    atomic temp+rename.

    Returns (n_lines, n_substituted, n_passthrough, n_parse_errors).
    """
    n_lines = 0
    n_substituted = 0
    n_passthrough = 0
    n_parse_errors = 0

    if dry_run:
        with source_path.open() as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                n_lines += 1
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    hash_id = record.get("hash_id")
                except json.JSONDecodeError:
                    n_parse_errors += 1
                    continue
                if hash_id in corrections_index:
                    if hash_id in substituted_hash_ids:
                        multiply_used_hash_ids.add(hash_id)
                    substituted_hash_ids.add(hash_id)
                    n_substituted += 1
                else:
                    n_passthrough += 1
        return n_lines, n_substituted, n_passthrough, n_parse_errors

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with source_path.open() as src, tmp_path.open("w") as dst:
        for raw in src:
            n_lines += 1
            stripped = raw.rstrip("\n")
            if not stripped:
                dst.write(raw)
                continue
            try:
                record = json.loads(stripped)
                hash_id = record.get("hash_id")
            except json.JSONDecodeError:
                # Preserve unparseable line as-is (should never occur on production data)
                dst.write(raw)
                n_parse_errors += 1
                continue
            if hash_id in corrections_index:
                if hash_id in substituted_hash_ids:
                    multiply_used_hash_ids.add(hash_id)
                substituted_hash_ids.add(hash_id)
                dst.write(corrections_index[hash_id] + "\n")
                n_substituted += 1
            else:
                # Byte-for-byte passthrough
                dst.write(raw)
                n_passthrough += 1
    os.replace(tmp_path, output_path)
    return n_lines, n_substituted, n_passthrough, n_parse_errors


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source-root", type=Path, required=True,
                    help="Read-only path to the source dataset tree (`llama405b_assessed/`).")
    ap.add_argument("--reinput-file", type=Path, required=True,
                    help="JSONL file with corrected records (output of the reinput-garbage sbatch).")
    ap.add_argument("--output-root", type=Path, required=True,
                    help="Path where the merged tree will be written. Must not equal "
                         "--source-root and must not be nested inside it.")
    ap.add_argument("--output-manifest", type=Path, required=True,
                    help="Path to write a JSON manifest summarizing the merge.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write anything, just report what would happen.")
    ap.add_argument("--force", action="store_true",
                    help="Allow --output-root to be a non-empty existing directory (overwrites).")
    args = ap.parse_args()

    assert_paths_safe(args.source_root, args.output_root, args.reinput_file)

    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force and not args.dry_run:
        sys.exit(f"ERROR: --output-root exists and is non-empty: {args.output_root}\n"
                 f"  Use --force to overwrite, or pick a different path, or remove it first.")

    print(f"[merge_reinput] building corrections index from {args.reinput_file} ...", file=sys.stderr)
    corrections_index = build_corrections_index(args.reinput_file)
    print(f"[merge_reinput] indexed {len(corrections_index)} hash_ids from reinput file", file=sys.stderr)

    delta_files = sorted(args.source_root.glob("snapshot_*-01-01/llm_assessed/delta_*.jsonl"))
    print(f"[merge_reinput] found {len(delta_files)} source delta files", file=sys.stderr)
    if not delta_files:
        sys.exit(f"ERROR: no delta_*.jsonl files found under {args.source_root}/snapshot_*-01-01/llm_assessed/")

    substituted_hash_ids = set()    # hash_ids actually used in substitutions
    multiply_used_hash_ids = set()  # hash_ids substituted into >1 source record (data integrity warning)

    manifest_files = []
    grand_n_lines = 0
    grand_n_substituted = 0
    grand_n_passthrough = 0
    grand_n_parse_errors = 0

    for src in delta_files:
        rel = src.relative_to(args.source_root)
        dst = args.output_root / rel
        n_lines, n_sub, n_pass, n_err = merge_delta_file(
            src, dst, corrections_index, substituted_hash_ids, multiply_used_hash_ids, args.dry_run)
        grand_n_lines += n_lines
        grand_n_substituted += n_sub
        grand_n_passthrough += n_pass
        grand_n_parse_errors += n_err
        marker = "(dry-run)" if args.dry_run else "wrote"
        print(f"[merge_reinput] {rel}: lines={n_lines}, substituted={n_sub}, passthrough={n_pass}"
              + (f", parse_errors={n_err}" if n_err else "")
              + f"  → {marker}: {dst}",
              file=sys.stderr)
        manifest_files.append({
            "rel_path": str(rel),
            "n_lines": n_lines,
            "n_substituted": n_sub,
            "n_passthrough": n_pass,
            "n_parse_errors": n_err,
        })

    unmatched = set(corrections_index.keys()) - substituted_hash_ids

    manifest = {
        "source_root": str(args.source_root.resolve()),
        "reinput_file": str(args.reinput_file.resolve()),
        "output_root": str(args.output_root.resolve()),
        "dry_run": args.dry_run,
        "n_correction_records": len(corrections_index),
        "n_correction_records_matched_in_source": len(substituted_hash_ids),
        "n_correction_records_unmatched": len(unmatched),
        "n_hash_ids_substituted_multiple_times": len(multiply_used_hash_ids),
        "n_source_delta_files": len(delta_files),
        "n_source_lines_total": grand_n_lines,
        "n_lines_substituted_total": grand_n_substituted,
        "n_lines_passthrough_total": grand_n_passthrough,
        "n_source_parse_errors": grand_n_parse_errors,
        "files": manifest_files,
        "sample_unmatched_hash_ids": sorted(unmatched)[:10],
        "sample_multiply_used_hash_ids": sorted(multiply_used_hash_ids)[:10],
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    print(file=sys.stderr)
    print("[merge_reinput] === SUMMARY ===", file=sys.stderr)
    print(f"  source-root:                {args.source_root}", file=sys.stderr)
    print(f"  reinput-file:               {args.reinput_file}", file=sys.stderr)
    print(f"  output-root:                {args.output_root}"
          + (" (DRY RUN — nothing written)" if args.dry_run else ""),
          file=sys.stderr)
    print(f"  correction records:         {len(corrections_index)}", file=sys.stderr)
    print(f"  matched in source:          {len(substituted_hash_ids)}", file=sys.stderr)
    print(f"  UNMATCHED (orphans):        {len(unmatched)}", file=sys.stderr)
    if multiply_used_hash_ids:
        print(f"  WARNING — substituted multiple times: {len(multiply_used_hash_ids)} hash_ids", file=sys.stderr)
        print(f"  (this implies duplicate hash_ids in source — data integrity issue)", file=sys.stderr)
    print(f"  source delta files:         {len(delta_files)}", file=sys.stderr)
    print(f"  total source lines:         {grand_n_lines}", file=sys.stderr)
    print(f"  lines substituted:          {grand_n_substituted}", file=sys.stderr)
    print(f"  lines pass-through:         {grand_n_passthrough}", file=sys.stderr)
    if grand_n_parse_errors:
        print(f"  source parse errors:        {grand_n_parse_errors}", file=sys.stderr)
    print(f"  manifest:                   {args.output_manifest}", file=sys.stderr)


if __name__ == "__main__":
    main()
