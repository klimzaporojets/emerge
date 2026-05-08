#!/usr/bin/env python3
"""Identify garbage clusters in the 405B-v1 curation annotations, by file
and line-range. Companion to `spot_check_405b_assessments.py`.

The goal is to pinpoint contiguous-line ranges within delta JSONL files
where TGI output went degenerate (loops, truncation, repeated tokens).
The hypothesis — confirmed by sample-level spot-check — is that garbage
correlates with TGI fault-recovery windows: when TGI recovers from a
fault but is briefly producing degenerate output before fully stabilizing,
all records processed during that window get bad annotations. These
windows show up as line_idx clusters within a single delta file.

What this script does:
  1. Streams every `llm_assessed/snapshot_*-01-01/llm_assessed/delta_*.jsonl`
     under the --root directory.
  2. For each triple's `Meta-Llama-3.1-405B_prompt_v1` assessment, runs
     the same garbage-detection heuristics as `spot_check_405b_assessments.py`
     on the explanation text stored in the `llm_prompt` field.
  3. Filters to "serious" flags only: repeated_token, repeated_ngram,
     low_unique_ratio, truncation, empty, very_few_words.
     Skips `repeated_punct` (heuristic FP — usually triggered by passage
     content like exclamation-mark-laden anime titles, not LLM garbage).
     Skips `contradiction` by default (5,368 cases mostly phrasing FPs;
     pass --include-contradiction to add them).
  4. Groups flagged triples by (year, delta_file). Within each file,
     sorts by line_idx and applies sliding-window clustering: a "cluster"
     is a maximal set where each adjacent flagged line is within
     --cluster-gap (default 50) of the next.
  5. Pads each cluster ±--cluster-pad (default 20) lines on each side
     to catch sub-threshold degeneration in the tail of the bad TGI window.
  6. Outputs:
        --output-md     human-readable report
        --output-jsonl  machine-readable cluster definitions for the
                        downstream reinput script

JSONL schema for each cluster:
    {
      "year": "2024",
      "delta_file": "delta_2024-01-08.jsonl",
      "line_start": 71,           # first flagged line_idx in this cluster
      "line_end":   694,          # last flagged line_idx
      "padded_start": 51,         # = max(0, line_start - pad)
      "padded_end":   714,        # = line_end + pad
      "n_flagged_lines": 23,      # count of distinct line_idx with at least one flagged triple
      "n_flagged_triples": 47,    # count of individual triples flagged
      "span": 643,                # = line_end - line_start
      "density": 0.0358,          # = n_flagged_lines / (span + 1)
      "flag_counts": {            # per-flag breakdown
        "repeated_token": 14,
        "truncation": 21,
        ...
      },
      "flagged_lines": [71, 177, 211, 277, ...]
    }

Usage (read-only on any 405B-style annotation tree, e.g. the output of
your own `s05_generate_dataset_with_llm_v9_llama405b.sh` run):

    python scripts/dataset/find_garbage_clusters.py \\
        --root <your-output-tree>/llama405b_assessed \\
        --output-md output/garbage_clusters_$(date +%Y%m%d_%H%M).md \\
        --output-jsonl output/garbage_clusters_$(date +%Y%m%d_%H%M).jsonl

The Markdown is for you to eyeball before committing to a reinput run.
The JSONL feeds `build_reinput_for_garbage.py` (separate script, written
after you review the clusters).
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


LLM_NAME_405B_V1 = "Meta-Llama-3.1-405B_prompt_v1"

# Flags we treat as REAL garbage (must trigger re-assessment). These are
# the failure modes where, on a manual spot-check, we observed the LLM
# verdict was often wrong (false NO from a "Yes Yes Yes..." loop misparsed).
SERIOUS_FLAGS = {
    "repeated_token",
    "repeated_ngram",
    "low_unique_ratio",
    "truncation",
    "empty",
    "very_few_words",
}

# Heuristic false-positive — usually triggered by passage content (e.g.,
# titles like "Teppen—!!!!!!!!!!!!!!!"), not LLM-generated garbage.
HEURISTIC_FP_FLAGS = {"repeated_punct"}

# Defer-to-manual-review — ~5K cases, many are phrasing artifacts
# ("YES because there is no..."). Set --include-contradiction to add.
DEFERRED_FLAGS = {"contradiction"}


def detect_garbage(text):
    """Same heuristics as spot_check_405b_assessments.py:detect_garbage,
    duplicated here so this script is self-contained.
    """
    flags = []
    if text is None or text.strip() == "":
        flags.append("empty")
        return flags
    words = re.findall(r"\w+", text.lower())
    n_words = len(words)
    n_unique = len(set(words)) if n_words else 0
    head = text.strip()[:60].upper()

    if n_words < 3:
        flags.append("very_few_words")

    if "YES" in head and "NO" in head:
        flags.append("contradiction")
    if n_words >= 4 and n_unique / n_words < 0.30:
        flags.append("low_unique_ratio")

    if n_words >= 6:
        ngrams = Counter(tuple(words[i : i + 3]) for i in range(n_words - 2))
        max_count = max(ngrams.values()) if ngrams else 0
        if max_count > 5:
            flags.append("repeated_ngram")

    raw_tokens = re.findall(r"\S+", text)
    consec = 1
    max_consec_run = 1
    last = None
    for tok in raw_tokens:
        if tok == last:
            consec += 1
            if consec > max_consec_run:
                max_consec_run = consec
        else:
            consec = 1
        last = tok
    if max_consec_run > 4:
        flags.append("repeated_token")

    if re.search(r"([^\w\s])\1{7,}", text):
        flags.append("repeated_punct")

    if n_words > 1900:
        flags.append("truncation")
    elif n_words > 600 and not any(f in flags for f in
            ("low_unique_ratio", "repeated_ngram", "repeated_token", "repeated_punct")):
        flags.append("very_long")

    return flags


def get_explanation(assessment):
    """Pull the LLM explanation from the assessment record. Field name
    in the production v8 pipeline is `llm_prompt` (misnamed; it stores
    the response explanation).
    """
    for k in ("llm_prompt", "llm_response_text", "response_text", "raw_response", "explanation", "llm_explanation"):
        v = assessment.get(k)
        if v:
            return v
    return None


def cluster_lines(line_idxs, gap):
    """Group sorted line_idxs into clusters where each adjacent pair is
    within `gap` of each other. Returns list of (start, end, idxs)."""
    if not line_idxs:
        return []
    line_idxs = sorted(set(line_idxs))
    clusters = []
    cur = [line_idxs[0]]
    for idx in line_idxs[1:]:
        if idx - cur[-1] <= gap:
            cur.append(idx)
        else:
            clusters.append(cur)
            cur = [idx]
    clusters.append(cur)
    return [(c[0], c[-1], c) for c in clusters]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", type=Path, required=True,
                    help="Path to the llama405b_assessed/ directory.")
    ap.add_argument("--cluster-gap", type=int, default=50,
                    help="Max line_idx gap between adjacent flagged lines "
                         "to group them into the same cluster. Default 50.")
    ap.add_argument("--cluster-pad", type=int, default=20,
                    help="Lines to pad on each side of each cluster (to "
                         "catch sub-threshold degeneration). Default 20.")
    ap.add_argument("--include-contradiction", action="store_true",
                    help="Also treat `contradiction` as serious. Default off "
                         "(too many cases, mostly phrasing FPs; recommend "
                         "manual sample first).")
    ap.add_argument("--output-md", type=Path, required=True,
                    help="Markdown report path.")
    ap.add_argument("--output-jsonl", type=Path, required=True,
                    help="Machine-readable cluster JSONL path.")
    args = ap.parse_args()

    serious = set(SERIOUS_FLAGS)
    if args.include_contradiction:
        serious.add("contradiction")

    # (year, delta_file) → list of (line_idx, triple_qids, flags)
    flagged_by_file = defaultdict(list)
    n_total = 0
    n_flagged_triples = 0
    n_total_lines_seen = defaultdict(int)  # (year, delta_file) → max line_idx seen

    print(f"[find_garbage_clusters] scanning {args.root} ...", file=sys.stderr)
    for snap_dir in sorted(args.root.glob("snapshot_*-01-01")):
        year = snap_dir.name.split("_")[1].split("-")[0]
        la_dir = snap_dir / "llm_assessed"
        if not la_dir.is_dir():
            continue
        for delta_file in sorted(la_dir.glob(f"delta_{year}-*.jsonl")):
            delta_name = delta_file.name
            with delta_file.open() as fh:
                for line_idx, line in enumerate(fh):
                    n_total_lines_seen[(year, delta_name)] = line_idx
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for triple in record.get("tkgu_triples", []):
                        for a in triple.get("llm_assessment", []):
                            if a.get("llm_name") != LLM_NAME_405B_V1:
                                continue
                            n_total += 1
                            explanation = get_explanation(a)
                            flags = detect_garbage(explanation)
                            serious_hits = [f for f in flags if f in serious]
                            if not serious_hits:
                                continue
                            triple_qids = tuple(triple.get("triple") or [None, None, None])
                            flagged_by_file[(year, delta_name)].append(
                                (line_idx, triple_qids, serious_hits)
                            )
                            n_flagged_triples += 1
                    if n_total % 50000 == 0:
                        print(f"[find_garbage_clusters]   scanned {n_total:,} ...", file=sys.stderr)

    print(f"[find_garbage_clusters] done. total assessments={n_total:,}, "
          f"flagged triples (serious)={n_flagged_triples:,} "
          f"({100*n_flagged_triples/max(n_total,1):.4f}%)", file=sys.stderr)

    # Per-file cluster detection.
    cluster_records = []   # the JSONL rows
    file_summaries = []    # (year, delta, total_flagged_triples, n_clusters, top_cluster_span)

    for (year, delta_name), entries in sorted(flagged_by_file.items()):
        flagged_lines = [e[0] for e in entries]
        clusters = cluster_lines(flagged_lines, args.cluster_gap)

        for (line_start, line_end, line_set) in clusters:
            entries_in = [e for e in entries if e[0] in set(line_set)]
            flag_counts = Counter()
            for _, _, fs in entries_in:
                for f in fs:
                    flag_counts[f] += 1
            cluster_records.append({
                "year": year,
                "delta_file": delta_name,
                "line_start": line_start,
                "line_end": line_end,
                "padded_start": max(0, line_start - args.cluster_pad),
                "padded_end": line_end + args.cluster_pad,
                "n_flagged_lines": len(line_set),
                "n_flagged_triples": len(entries_in),
                "span": line_end - line_start,
                "density": len(line_set) / max(1, line_end - line_start + 1),
                "flag_counts": dict(flag_counts),
                "flagged_lines": list(line_set),
            })
        file_summaries.append((
            year, delta_name, len(entries), len(clusters),
            max((c[1] - c[0] for c in clusters), default=0),
        ))

    # ---------- write JSONL ----------
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as fh:
        for rec in cluster_records:
            fh.write(json.dumps(rec) + "\n")
    print(f"[find_garbage_clusters] wrote {args.output_jsonl} "
          f"({len(cluster_records)} clusters)", file=sys.stderr)

    # ---------- write Markdown ----------
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_md.open("w") as out:
        out.write("# 405B-v1 Garbage Cluster Report\n\n")
        out.write(f"Source: `{args.root}`  \n")
        out.write(f"Cluster gap: `{args.cluster_gap}` lines  \n")
        out.write(f"Cluster pad: `±{args.cluster_pad}` lines  \n")
        out.write(f"Serious flags: `{sorted(serious)}`  \n")
        out.write(f"\nTotal `405B-v1` assessments scanned: **{n_total:,}**  \n")
        out.write(f"Total triples flagged with serious garbage: "
                  f"**{n_flagged_triples:,}** "
                  f"({100*n_flagged_triples/max(n_total,1):.4f}%)  \n\n")

        out.write("## Per-file summary (sorted by flagged-triple count, descending)\n\n")
        out.write("| Year | Delta file | Flagged triples | # clusters | Largest cluster span |\n")
        out.write("|---|---|---:|---:|---:|\n")
        for year, delta, n_trip, n_clu, span in sorted(
                file_summaries, key=lambda r: -r[2]):
            if n_trip == 0:
                continue
            out.write(f"| {year} | `{delta}` | {n_trip:,} | {n_clu} | {span} |\n")
        out.write("\n")

        out.write("## Per-cluster detail (sorted by flagged-triple count, descending)\n\n")
        for rec in sorted(cluster_records, key=lambda r: -r["n_flagged_triples"]):
            out.write(f"### `{rec['year']} / {rec['delta_file']}` "
                      f"lines {rec['line_start']}–{rec['line_end']} "
                      f"(span {rec['span']}, padded → {rec['padded_start']}–{rec['padded_end']})\n\n")
            out.write(f"- flagged triples: **{rec['n_flagged_triples']:,}** "
                      f"across {rec['n_flagged_lines']:,} distinct lines\n")
            out.write(f"- density: {rec['density']:.4f} flagged-lines per line in the span\n")
            out.write(f"- flag breakdown: ")
            out.write(", ".join(f"`{f}`={c}" for f, c in
                                sorted(rec["flag_counts"].items(),
                                       key=lambda kv: -kv[1])))
            out.write("\n")
            out.write(f"- flagged line_idx values (first 50 shown): "
                      f"`{rec['flagged_lines'][:50]}`")
            if len(rec["flagged_lines"]) > 50:
                out.write(f" *(+{len(rec['flagged_lines'])-50} more)*")
            out.write("\n\n---\n\n")

        out.write("## Recommended next step\n\n")
        out.write(f"Total clusters: **{len(cluster_records)}**.  \n")
        total_padded_lines = sum(r['padded_end'] - r['padded_start'] + 1
                                 for r in cluster_records)
        out.write(f"Total line-range coverage (padded): "
                  f"**{total_padded_lines:,} line-records** to re-query.  \n\n")
        out.write("Feed this JSONL to `build_reinput_for_garbage.py` (next "
                  "script) which will: (a) read the original llm_assessed "
                  "JSONLs, (b) within each padded cluster range, delete the "
                  "`Meta-Llama-3.1-405B_prompt_v1` entries from "
                  "`llm_assessment` so the existing per-triple skip logic "
                  "will re-query them, (c) write a new reinput tree for a "
                  "small sbatch.\n")

    print(f"[find_garbage_clusters] wrote {args.output_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
