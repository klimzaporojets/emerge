#!/usr/bin/env python3
"""Inspect lines IMMEDIATELY ADJACENT to garbage clusters, to validate
how tight the cluster-detection heuristic actually was.

Companion to `find_garbage_clusters.py`. The motivating question:

    For a cluster spanning lines [start, end] in a delta file, is the bad
    TGI window EXACTLY [start, end], or does it leak into lines slightly
    outside (which our heuristics didn't catch)?

If neighbor lines (start-1, end+1, etc.) show LLM verdicts that make
sense given their passages, the heuristic captured the boundary tightly
→ Option B-light (re-query only flagged lines) is sufficient.

If neighbor lines show suspect verdicts despite clean-looking
explanations, degeneration leaked → need Option C (re-query the full
padded range) or larger padding.

What this script does:
  1. Reads the cluster JSONL produced by find_garbage_clusters.py.
  2. For each cluster, picks N "neighbor offsets" (e.g., -10, -5, -1, +1,
     +5, +10) relative to line_start / line_end.
  3. For each (delta_file, neighbor_line_idx), reads the actual record
     from the source llm_assessed JSONL and dumps:
       - passage text
       - each tkgu_triple with its Meta-Llama-3.1-405B_prompt_v1 verdict
         and explanation
  4. Writes Markdown for human review.

Usage:

    python scripts/dataset/inspect_cluster_neighbors.py \\
        --root <your-output-tree>/llama405b_assessed \\
        --clusters-jsonl output/garbage_clusters_<ts>.jsonl \\
        --neighbors -10,-5,-1,+1,+5,+10 \\
        --n-clusters 8 \\
        --mode random \\
        --seed 42 \\
        --output output/cluster_neighbors_$(date +%Y%m%d_%H%M).md

`--n-clusters` and `--neighbors` are hyperparameters. Defaults:
  - 8 clusters × 6 neighbors = 48 lines = a few hundred triple verdicts —
    enough to be representative without overwhelming the human reviewer.
  - The largest cluster is ALWAYS included (it dominates volume — 67% of
    flagged triples in our case live in one cluster, so skipping it would
    waste the diagnostic).
  - Remaining N-1 are randomly sampled (default `--mode random`) so we
    cover small/singleton clusters fairly, not just the top-N tail.
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


LLM_NAME_405B_V1 = "Meta-Llama-3.1-405B_prompt_v1"


def get_explanation(assessment):
    for k in ("llm_prompt", "llm_response_text", "response_text", "raw_response", "explanation", "llm_explanation"):
        v = assessment.get(k)
        if v:
            return v
    return None


def short_triple(triple):
    labels = triple.get("triple_labels") or [None, None, None]
    qids = triple.get("triple") or [None, None, None]
    parts = [labels[i] or qids[i] or "?" for i in (0, 1, 2)]
    return f"{parts[0]}  --[{parts[1]}]-->  {parts[2]}"


def parse_neighbors_arg(s):
    """Parse '-10,-5,-1,+1,+5,+10' → [-10, -5, -1, 1, 5, 10]."""
    return [int(x) for x in s.split(",")]


def select_clusters(clusters, n, mode, seed):
    """Pick which clusters to inspect.

    Modes:
      - "top": deterministic — top N by flagged-triple count.
      - "random": always include the largest cluster (since one mega-
        cluster typically dominates and skipping it would waste the
        diagnostic), then randomly sample N-1 from the rest. Seeded for
        reproducibility.
      - "stratified": include the largest, then split the remaining
        clusters into roughly-equal-size buckets by flagged-triple
        count (large / medium / small / singleton) and sample evenly.

    Returns clusters in display order: largest first, then chosen
    samples in original cluster order.
    """
    sorted_by_size = sorted(clusters, key=lambda c: -c["n_flagged_triples"])
    if not sorted_by_size:
        return []
    n = min(n, len(sorted_by_size))
    if mode == "top":
        return sorted_by_size[:n]

    rng = random.Random(seed)
    largest = sorted_by_size[0]
    rest = sorted_by_size[1:]
    if n == 1 or not rest:
        return [largest]

    if mode == "random":
        sampled = rng.sample(rest, k=n - 1)
        return [largest] + sampled

    # stratified
    # Bucket by ranges of flagged triples: pick edges so each bucket
    # is non-empty if data permits.
    counts = sorted({c["n_flagged_triples"] for c in rest})
    if len(counts) >= 4:
        q = [counts[len(counts) * k // 4] for k in (1, 2, 3)]
    else:
        q = counts[:3] + [10**9] * max(0, 3 - len(counts))
    buckets = [[], [], [], []]  # singleton/small / small / medium / large
    for c in rest:
        n_ft = c["n_flagged_triples"]
        if n_ft <= q[0]:
            buckets[0].append(c)
        elif n_ft <= q[1]:
            buckets[1].append(c)
        elif n_ft <= q[2]:
            buckets[2].append(c)
        else:
            buckets[3].append(c)
    per_bucket = max(1, (n - 1) // 4)
    sampled = []
    for b in buckets:
        if not b:
            continue
        k = min(per_bucket, len(b))
        sampled.extend(rng.sample(b, k=k))
    # If we under-shot due to small/empty buckets, top up randomly.
    remaining_needed = (n - 1) - len(sampled)
    if remaining_needed > 0:
        leftover = [c for c in rest if c not in sampled]
        if leftover:
            sampled.extend(rng.sample(leftover, k=min(remaining_needed, len(leftover))))
    return [largest] + sampled


def collect_target_lines(selected_clusters, neighbor_offsets):
    """For each (already-selected) cluster, compute the list of
    (delta_file, year, line_idx, role, cluster) tuples we want to
    inspect.
    """
    targets = []
    for c in selected_clusters:
        for off in neighbor_offsets:
            if off < 0:
                target_line = c["line_start"] + off
                role = f"BEFORE start (offset {off:+d})"
            else:
                target_line = c["line_end"] + off
                role = f"AFTER end (offset +{off})"
            if target_line < 0:
                continue
            targets.append((c["year"], c["delta_file"], target_line, role, c))
    return targets


def fetch_records(root, targets):
    """Read each target line from its delta file. Returns dict
    {(year, delta_file, line_idx): record_or_None}."""
    by_file = defaultdict(set)
    for year, delta_file, line_idx, _, _ in targets:
        by_file[(year, delta_file)].add(line_idx)
    records = {}
    for (year, delta_file), wanted_lines in by_file.items():
        delta_path = root / f"snapshot_{year}-01-01" / "llm_assessed" / delta_file
        if not delta_path.is_file():
            for ln in wanted_lines:
                records[(year, delta_file, ln)] = None
            continue
        max_wanted = max(wanted_lines)
        with delta_path.open() as fh:
            for line_idx, line in enumerate(fh):
                if line_idx > max_wanted:
                    break
                if line_idx not in wanted_lines:
                    continue
                try:
                    records[(year, delta_file, line_idx)] = json.loads(line)
                except json.JSONDecodeError:
                    records[(year, delta_file, line_idx)] = None
        for ln in wanted_lines:
            records.setdefault((year, delta_file, ln), None)
    return records


def render_record(out, year, delta_file, line_idx, role, cluster, record):
    out.write(f"### Neighbor: `{year} / {delta_file} / line {line_idx}`\n\n")
    out.write(f"- role: **{role}** (cluster lines {cluster['line_start']}–{cluster['line_end']}, "
              f"{cluster['n_flagged_triples']} flagged triples)\n")
    if record is None:
        out.write("- _(line not found in delta file — likely past the end-of-file "
                  "or the line wasn't yet processed when the snapshot was taken)_\n\n---\n\n")
        return

    chunk = record.get("chunk") or record.get("passage") or ""
    chunk_short = chunk if len(chunk) <= 1500 else (chunk[:1500] + "  …(truncated)")
    out.write(f"\n**Passage:**\n\n> {chunk_short}\n\n")

    triples = record.get("tkgu_triples", [])
    out.write(f"**405B-v1 assessments on this line ({len(triples)} triple slot(s)):**\n\n")
    if not triples:
        out.write("_(no triples on this line — possibly a fully-deduplicated record)_\n\n")
    else:
        for i, triple in enumerate(triples, 1):
            v1 = next((a for a in triple.get("llm_assessment", [])
                       if a.get("llm_name") == LLM_NAME_405B_V1), None)
            if v1 is None:
                out.write(f"#### Triple {i}: `{short_triple(triple)}`\n\n"
                          f"- prompt_type: `?`\n"
                          f"- 405B-v1 verdict: **NOT ASSESSED**\n\n")
                continue
            verdict = v1.get("llm_assessment")
            prompt_type = v1.get("llm_prompt_type", "?")
            explanation = get_explanation(v1) or "_(no explanation stored)_"
            exp_short = explanation if len(explanation) <= 1200 else (explanation[:1200] + "  …(truncated)")
            out.write(f"#### Triple {i}: `{short_triple(triple)}`\n\n")
            out.write(f"- prompt_type: `{prompt_type}`\n")
            out.write(f"- 405B-v1 verdict: **`{verdict}`**\n\n")
            out.write(f"**Explanation:**\n\n```\n{exp_short}\n```\n\n")
            out.write(f"**Reviewer**: agree/disagree with `{verdict}` here? "
                      f"(Looking for: does it make sense, OR is it subtly wrong "
                      f"despite a clean explanation?)\n\n")
        out.write("\n")
    out.write("---\n\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", type=Path, required=True,
                    help="llama405b_assessed/ root (same as find_garbage_clusters --root).")
    ap.add_argument("--clusters-jsonl", type=Path, required=True,
                    help="JSONL output from find_garbage_clusters.py.")
    ap.add_argument("--neighbors", type=str, default="-10,-5,-1,+1,+5,+10",
                    help="Comma-separated offsets relative to cluster start/end. "
                         "Negative → before line_start, positive → after line_end. "
                         "Default 6 offsets × N clusters keeps the report at a "
                         "manageable size for human review.")
    ap.add_argument("--n-clusters", type=int, default=8,
                    help="Number of clusters to inspect (hyperparameter — "
                         "default 8 ≈ 48 lines = a few hundred triple verdicts, "
                         "enough to be representative without overwhelming the "
                         "reviewer).")
    ap.add_argument("--mode", choices=("top", "random", "stratified"),
                    default="random",
                    help="How to pick clusters. 'top' = deterministic top-N by "
                         "size. 'random' (default) = always include the "
                         "largest cluster (which dominates volume), randomly "
                         "sample the remaining N-1. 'stratified' = bucket "
                         "remaining clusters by flagged-triple count and "
                         "sample across buckets so we cover both big "
                         "clusters and singletons.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for --mode random / stratified — change "
                         "to re-roll the sample, keep fixed for reproducibility.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output Markdown path.")
    args = ap.parse_args()

    neighbor_offsets = parse_neighbors_arg(args.neighbors)

    with args.clusters_jsonl.open() as fh:
        clusters = [json.loads(line) for line in fh if line.strip()]

    selected = select_clusters(clusters, args.n_clusters, args.mode, args.seed)
    targets = collect_target_lines(selected, neighbor_offsets)
    print(f"[inspect_cluster_neighbors] {len(clusters)} clusters total, "
          f"selected {len(selected)} via mode={args.mode} (seed={args.seed}) → "
          f"{len(targets)} neighbor lines to read")
    for c in selected:
        print(f"  - {c['year']}/{c['delta_file']} lines "
              f"{c['line_start']}-{c['line_end']} "
              f"({c['n_flagged_triples']} flagged)")

    records = fetch_records(args.root, targets)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        out.write("# Cluster-Neighbor Inspection Report\n\n")
        out.write(f"Source clusters: `{args.clusters_jsonl}`  \n")
        out.write(f"Source data: `{args.root}`  \n")
        out.write(f"Neighbor offsets: `{neighbor_offsets}`  \n")
        out.write(f"Cluster selection: `mode={args.mode}`, "
                  f"`n={args.n_clusters}`, `seed={args.seed}` "
                  f"({len(selected)} of {len(clusters)} clusters chosen — "
                  f"largest is always included, others sampled per the mode).\n\n")
        out.write("**How to read this report**: for each cluster, neighbors at "
                  "+/- 1, 5, 10 lines beyond the cluster boundary are dumped. "
                  "Read each neighbor's passage + 405B-v1 verdict on each "
                  "triple. If verdicts make sense given the passage, the "
                  "cluster boundary is tight and Option B-light is "
                  "sufficient. If verdicts look subtly wrong (e.g., clear "
                  "support in passage but verdict=False, or clear lack of "
                  "support but verdict=True), the bad TGI window leaked beyond "
                  "the heuristic boundary and we need Option C / larger "
                  "padding.\n\n")

        for c in selected:
            out.write(f"## Cluster: `{c['year']} / {c['delta_file']}` "
                      f"lines {c['line_start']}–{c['line_end']}  "
                      f"({c['n_flagged_triples']} flagged triples)\n\n")
            out.write(f"flag breakdown: ")
            out.write(", ".join(f"`{f}`={n}" for f, n in
                                sorted(c['flag_counts'].items(),
                                       key=lambda kv: -kv[1])))
            out.write("\n\n")
            for off in neighbor_offsets:
                if off < 0:
                    target_line = c["line_start"] + off
                    role = f"BEFORE start (offset {off:+d})"
                else:
                    target_line = c["line_end"] + off
                    role = f"AFTER end (offset +{off})"
                if target_line < 0:
                    continue
                rec = records.get((c["year"], c["delta_file"], target_line))
                render_record(out, c["year"], c["delta_file"], target_line, role, c, rec)

    print(f"[inspect_cluster_neighbors] wrote {args.output}")


if __name__ == "__main__":
    main()
