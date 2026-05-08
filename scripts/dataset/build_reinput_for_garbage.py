#!/usr/bin/env python3
"""Build a re-input tree for the garbage 405B-v1 assessments, so a small
sbatch can re-query ONLY the affected triples.

Companion to `find_garbage_clusters.py` (which produced the cluster JSONL)
and `inspect_cluster_neighbors.py` (which validated cluster boundaries
are tight, justifying Option B-light: re-query only triples on flagged
lines).

What this script does — and explicitly does NOT do:

  - DOES: read the SOURCE 405B-v1 annotation tree at --source-root.
  - DOES: for each delta file that has flagged lines per the cluster
    JSONL, write a MODIFIED COPY of that delta file to --output-root,
    with the `Meta-Llama-3.1-405B_prompt_v1` entry removed from each
    triple's `llm_assessment` list ON FLAGGED LINES ONLY.
  - DOES: skip delta files with no flagged lines — they will not appear
    in the output tree at all.
  - DOES NOT: modify, overwrite, rename, or delete anything under
    --source-root. The source tree stays byte-for-byte identical.
  - DOES NOT: copy/symlink/hardlink unaffected source files. The output
    tree is sparse — only the ~29 modified delta files, not the full
    ~35-file source tree.

After running this:
  - Source tree (e.g., `20260502_v8_llama_405b_v1_complete_dataset/`)
    is untouched.
  - Output tree (e.g., `_complete_dataset_reinput_garbage_1/`) contains
    only the modified delta files, ready to feed a small re-query
    sbatch. The pipeline's per-triple-skip logic will see "no 405B-v1
    assessment" on the stripped triples, call TGI for them, and write
    the new assessments back into THE OUTPUT TREE (still not source).

Usage:

    python scripts/dataset/build_reinput_for_garbage.py \\
        --source-root <your-output-tree>/llama405b_assessed \\
        --output-root <reinput-tree>/llama405b_assessed \\
        --clusters-jsonl output/garbage_clusters_<ts>.jsonl \\
        --output-manifest output/reinput_manifest_$(date +%Y%m%d_%H%M).json

Add `--dry-run` to see what would happen without writing anything.
Add `--force` to overwrite a non-empty --output-root (off by default).
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_TARGET_LLM = "Meta-Llama-3.1-405B_prompt_v1"


def is_subpath(child, parent):
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_paths_safe(source_root, output_root):
    if not source_root.is_dir():
        sys.exit(f"ERROR: --source-root does not exist or is not a directory: {source_root}")
    if source_root.resolve() == output_root.resolve():
        sys.exit(f"ERROR: --source-root and --output-root resolve to the same path: {source_root}")
    if is_subpath(output_root, source_root):
        sys.exit(f"ERROR: --output-root is inside --source-root. This is unsafe; pick a "
                 f"path outside the source tree.\n  source = {source_root}\n  output = {output_root}")
    if is_subpath(source_root, output_root):
        sys.exit(f"ERROR: --source-root is inside --output-root. This is unsafe.\n"
                 f"  source = {source_root}\n  output = {output_root}")


def load_flagged_lines_by_file(clusters_jsonl):
    """Returns dict {(year, delta_file): set(line_idx)}."""
    flagged = defaultdict(set)
    n_clusters = 0
    n_lines_total = 0
    with clusters_jsonl.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            c = json.loads(raw)
            year = c["year"]
            delta_file = c["delta_file"]
            for ln in c["flagged_lines"]:
                flagged[(year, delta_file)].add(ln)
                n_lines_total += 1
            n_clusters += 1
    return flagged, n_clusters, n_lines_total


def strip_target_assessment(record, target_llm_name):
    """For each triple in record['tkgu_triples'], remove any
    llm_assessment entry whose llm_name == target_llm_name. Returns the
    number of triples whose assessment list was actually shortened.
    """
    n_stripped = 0
    for triple in record.get("tkgu_triples", []):
        before = triple.get("llm_assessment", [])
        if not before:
            continue
        after = [a for a in before if a.get("llm_name") != target_llm_name]
        if len(after) != len(before):
            triple["llm_assessment"] = after
            n_stripped += 1
    return n_stripped


def process_delta_file(source_path, output_path, flagged_line_set,
                       target_llm_name, dry_run):
    """Stream source_path; rewrite to output_path. On lines in
    flagged_line_set, strip target_llm_name from each triple's
    llm_assessment. Other lines pass through unchanged.

    Returns (n_lines_total, n_lines_modified, n_triples_stripped,
             n_lines_in_flagged_set_but_no_target).
    """
    n_lines_total = 0
    n_lines_modified = 0
    n_triples_stripped = 0
    n_flagged_no_target = 0   # flagged line but no target assessment found

    if dry_run:
        with source_path.open() as fh:
            for line_idx, line in enumerate(fh):
                n_lines_total += 1
                if line_idx not in flagged_line_set:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stripped = strip_target_assessment(record, target_llm_name)
                if stripped:
                    n_lines_modified += 1
                    n_triples_stripped += stripped
                else:
                    n_flagged_no_target += 1
        return n_lines_total, n_lines_modified, n_triples_stripped, n_flagged_no_target

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with source_path.open() as src, tmp_path.open("w") as dst:
        for line_idx, line in enumerate(src):
            n_lines_total += 1
            if line_idx not in flagged_line_set:
                dst.write(line)
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Preserve unparseable line as-is (this should never
                # happen on production data, but don't silently drop).
                dst.write(line)
                continue
            stripped = strip_target_assessment(record, target_llm_name)
            if stripped:
                n_lines_modified += 1
                n_triples_stripped += stripped
            else:
                n_flagged_no_target += 1
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_path)
    return n_lines_total, n_lines_modified, n_triples_stripped, n_flagged_no_target


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source-root", type=Path, required=True,
                    help="Read-only path to the existing 405B-v1 "
                         "annotation tree (the llama405b_assessed/ dir).")
    ap.add_argument("--output-root", type=Path, required=True,
                    help="Path where the modified copies will be written. "
                         "Must NOT be the same as --source-root and must "
                         "not be nested inside it. Will be created if "
                         "absent. If non-empty, --force is required.")
    ap.add_argument("--clusters-jsonl", type=Path, required=True,
                    help="Cluster JSONL produced by find_garbage_clusters.py.")
    ap.add_argument("--target-llm-name", type=str, default=DEFAULT_TARGET_LLM,
                    help=f"Name of the LLM whose assessment to strip "
                         f"(default: {DEFAULT_TARGET_LLM}).")
    ap.add_argument("--output-manifest", type=Path, required=True,
                    help="Path to write a JSON manifest summarizing what "
                         "was modified.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write anything, just report what would happen.")
    ap.add_argument("--force", action="store_true",
                    help="Allow --output-root to be a non-empty existing "
                         "directory (will overwrite existing delta files).")
    args = ap.parse_args()

    assert_paths_safe(args.source_root, args.output_root)

    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force and not args.dry_run:
        sys.exit(f"ERROR: --output-root exists and is non-empty: {args.output_root}\n"
                 f"  Use --force to overwrite, or pick a different path, or remove it first.")

    flagged_by_file, n_clusters, n_flagged_lines_total_with_dups = \
        load_flagged_lines_by_file(args.clusters_jsonl)
    print(f"[build_reinput] loaded {n_clusters} clusters → "
          f"{len(flagged_by_file)} distinct (year, delta_file) pairs, "
          f"{sum(len(s) for s in flagged_by_file.values())} unique flagged lines",
          file=sys.stderr)

    manifest_files = []
    grand_n_lines = 0
    grand_n_modified = 0
    grand_n_triples_stripped = 0
    grand_n_flagged_no_target = 0
    n_files_missing = 0

    for (year, delta_file), line_set in sorted(flagged_by_file.items()):
        # Auto-detect layout: nested (llm_assessed/) or flat (HF corpus).
        rel_nested = Path(f"snapshot_{year}-01-01") / "llm_assessed" / delta_file
        rel_flat = Path(f"snapshot_{year}-01-01") / delta_file
        if (args.source_root / rel_nested).is_file():
            rel_path = rel_nested
        elif (args.source_root / rel_flat).is_file():
            rel_path = rel_flat
        else:
            rel_path = rel_nested  # report the canonical "missing" path below
        src = args.source_root / rel_path
        dst = args.output_root / rel_path
        if not src.is_file():
            print(f"[build_reinput] MISSING in source: {src} — skipping", file=sys.stderr)
            n_files_missing += 1
            manifest_files.append({
                "year": year, "delta_file": delta_file,
                "rel_path": str(rel_path),
                "status": "MISSING_IN_SOURCE",
                "n_flagged_lines": len(line_set),
            })
            continue
        n_total, n_mod, n_stripped, n_no_target = process_delta_file(
            src, dst, line_set, args.target_llm_name, args.dry_run)
        grand_n_lines += n_total
        grand_n_modified += n_mod
        grand_n_triples_stripped += n_stripped
        grand_n_flagged_no_target += n_no_target
        marker = "(dry-run)" if args.dry_run else "wrote"
        print(f"[build_reinput] {year}/{delta_file}: "
              f"flagged={len(line_set)} lines, modified={n_mod}, "
              f"triples_stripped={n_stripped}"
              + (f", flagged_no_target={n_no_target}" if n_no_target else "")
              + f"  → {marker}: {dst}",
              file=sys.stderr)
        manifest_files.append({
            "year": year,
            "delta_file": delta_file,
            "rel_path": str(rel_path),
            "source_path": str(src),
            "output_path": str(dst),
            "n_flagged_lines_in_cluster_jsonl": len(line_set),
            "n_lines_in_source": n_total,
            "n_lines_modified": n_mod,
            "n_triples_stripped": n_stripped,
            "n_flagged_lines_with_no_target_assessment": n_no_target,
            "status": "DRY_RUN" if args.dry_run else "WRITTEN",
        })

    manifest = {
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "clusters_jsonl": str(args.clusters_jsonl.resolve()),
        "target_llm_name": args.target_llm_name,
        "dry_run": args.dry_run,
        "n_clusters": n_clusters,
        "n_distinct_files_in_clusters": len(flagged_by_file),
        "n_files_missing_in_source": n_files_missing,
        "n_lines_scanned_total": grand_n_lines,
        "n_lines_modified_total": grand_n_modified,
        "n_triples_stripped_total": grand_n_triples_stripped,
        "n_flagged_lines_with_no_target_assessment_total": grand_n_flagged_no_target,
        "files": manifest_files,
    }

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    print(file=sys.stderr)
    print(f"[build_reinput] === SUMMARY ===", file=sys.stderr)
    print(f"  source-root:               {args.source_root}", file=sys.stderr)
    print(f"  output-root:               {args.output_root}"
          + (" (DRY RUN — nothing written)" if args.dry_run else ""),
          file=sys.stderr)
    print(f"  clusters in JSONL:         {n_clusters}", file=sys.stderr)
    print(f"  distinct affected files:   {len(flagged_by_file)}", file=sys.stderr)
    print(f"  files missing in source:   {n_files_missing}", file=sys.stderr)
    print(f"  total lines scanned:       {grand_n_lines}", file=sys.stderr)
    print(f"  lines modified:            {grand_n_modified}", file=sys.stderr)
    print(f"  TRIPLES STRIPPED:          {grand_n_triples_stripped}", file=sys.stderr)
    print(f"  flagged lines w/o target:  {grand_n_flagged_no_target}", file=sys.stderr)
    print(f"  manifest:                  {args.output_manifest}", file=sys.stderr)


if __name__ == "__main__":
    main()
