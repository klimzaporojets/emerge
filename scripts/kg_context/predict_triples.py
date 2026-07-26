#!/usr/bin/env python3
"""PREDICT_TRIPLES — run the baseline(s) (KGGen / EDC+) on each COMPOSE_KG_PROMPT eval-set.

Fully config-driven via the compose config's `predict` block:
  "predict": {
    "baselines": ["kggen", "edc_plus"],   # choose kggen | edc_plus | both (any subset)
    "llm": "gpt-5.1",                      # which model (maps to a run_benchmarks config below)
    "points": null,                        # null = ALL composed points; or ["only_oracle","kg_rag_k8",...]
    "base_configs": {}                     # optional per-baseline config override {baseline: path}
  }
Reuses `benchmarks.run_benchmarks` UNCHANGED (CLI overrides input/output). Run in the BASELINE env
(`emerge`/`emerge-method`) with AZURE_OPENAI_API_KEY set (this step costs API $).

    PYTHONPATH=src python scripts/kg_context/predict_triples.py \
        --config config/kg_context/20260725_compose_only_oracle/config.json          # run
    PYTHONPATH=src python scripts/kg_context/predict_triples.py --config ... --dry-run  # preview cmds only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile


def _filtered_evalset(src_dir: str, allow: set, dst_dir: str) -> int:
    """Copy src_dir's eval-set (snapshot_*/delta_*.jsonl) into dst_dir keeping ONLY instances whose
    hash_id is in `allow`, preserving the dir structure so the wrapper reads it exactly like the full
    set. Returns #instances kept. Used by --hash-ids to run a SUBSET (e.g. 350) now; a later full run
    (no --hash-ids, SAME predict dir) skips these via obtain_already_predicted_hash_ids -> no LLM re-call."""
    kept = 0
    for fp in glob.glob(os.path.join(src_dir, "**", "*.jsonl"), recursive=True):
        rel = os.path.relpath(fp, src_dir)
        out = os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(fp) as fin, open(out, "w") as fout:
            for line in fin:
                line = line.strip()
                if line and json.loads(line).get("hash_id") in allow:
                    fout.write(line + "\n")
                    kept += 1
    return kept

# (baseline, llm) -> run_benchmarks config. gpt-5.1 = strongest baseline in the paper. Extend as needed.
# kg_prompt = our KG-conditioned baseline: its wrapper reads the composed `prompt` field VERBATIM
# (no passage rewriting), so predict_triples points run_benchmarks straight at the composed eval-set.
BASELINE_CONFIGS = {
    ("kggen", "gpt-5.1"):         "config/benchmarks/s02_run_benchmarks/20260116_kggen_gpt_5_1/config.json",
    ("kggen", "mistral-large"):   "config/benchmarks/s02_run_benchmarks/20260114_kggen_mistral_large/config.json",
    ("kggen", "mistral-small"):   "config/benchmarks/s02_run_benchmarks/20260114_kggen_mistral_small/config.json",
    ("edc_plus", "gpt-5.1"):      "config/benchmarks/s02_run_benchmarks/edc_plus_icl_gpt_5_1/config.json",
    ("edc_plus", "mistral-large"): "config/benchmarks/s02_run_benchmarks/edc_plus_icl_mistral_large/config.json",
    ("edc_plus", "mistral-small"): "config/benchmarks/s02_run_benchmarks/edc_plus_icl_mistral_small/config.json",
    ("kg_prompt", "gpt-5.1"):     "config/benchmarks/s02_run_benchmarks/kg_prompt_gpt_5_1/config.json",
}


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the COMPOSE_KG_PROMPT config (has output_dir + predict block)")
    ap.add_argument("--dry-run", action="store_true", help="print the run_benchmarks commands, do not execute")
    ap.add_argument("--hash-ids", help="file with hash_ids (one per line): predict ONLY these instances "
                    "(SUBSET run, e.g. 350). Re-run later WITHOUT it (same config -> same predict dir) to fill "
                    "the rest; the wrapper skips these via the shared output, so NO LLM re-call for the subset.")
    args = ap.parse_args()

    allow_ids: set | None = None
    if args.hash_ids:
        with open(args.hash_ids) as f:
            allow_ids = {ln.strip() for ln in f if ln.strip()}
        print(f"PREDICT_TRIPLES: SUBSET mode — restricting to {len(allow_ids)} hash_ids from {args.hash_ids}", flush=True)
    _tmp = tempfile.mkdtemp(prefix="predict_subset_") if allow_ids else None

    cfg = json.load(open(args.config))
    out_root = cfg["output_dir"]
    manifest_path = os.path.join(out_root, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"no manifest at {manifest_path} — run COMPOSE_KG_PROMPT first")
    manifest = json.load(open(manifest_path))

    pred = cfg["predict"]              # required config blocks/keys -> HARD FAIL (no silent 'gpt-5.1'/'kggen')
    llm = pred["llm"]
    baselines = pred["baselines"]
    overrides = pred["base_configs"] if "base_configs" in pred else {}   # optional per-baseline config override
    want_tags = pred["points"] if "points" in pred else None             # optional; None => all composed points
    pred_root = (pred["predict_output_dir"] if "predict_output_dir" in pred
                 else out_root.replace("/kg_prompts/", "/kg_prompts_pred/"))

    points = [p for p in manifest["points"] if (want_tags is None or p["tag"] in want_tags)]
    print(f"PREDICT_TRIPLES: baselines={baselines} llm={llm} | {len(points)} points "
          f"({[p['tag'] for p in points]}) -> {pred_root}", flush=True)

    n_ok = 0
    for p in points:
        # kg_prompt reads the composed `prompt` field directly, so point run_benchmarks straight at the
        # eval-set dir — no passage rewriting / throwaway model-input dir needed.
        model_in = p["input_dir"]
        if allow_ids is not None:   # SUBSET run: filter this point's eval-set to the allow-list (temp dir)
            model_in = os.path.join(_tmp, p["tag"])
            kept = _filtered_evalset(p["input_dir"], allow_ids, model_in)
            print(f"  [{p['tag']}] subset: {kept} instance(s) match the hash-id allow-list", flush=True)
        for bl in baselines:
            base_cfg = overrides[bl] if bl in overrides else BASELINE_CONFIGS.get((bl, llm))  # registry lookup handled below
            if not base_cfg:
                print(f"  !! no run_benchmarks config for ({bl}, {llm}); add it to predict.base_configs.{bl}",
                      flush=True)
                continue
            out_dir = os.path.join(pred_root, p["tag"], f"{bl}_{llm}")
            cmd = [sys.executable, "-u", "-m", "benchmarks.run_benchmarks",
                   "--config_file", base_cfg,
                   "--input_dataset_base_path", model_in + os.sep,
                   "--output_base_dir", out_dir + os.sep]
            print(f"  >>> [{bl}/{p['tag']}] " + " ".join(cmd), flush=True)
            if args.dry_run:
                continue
            os.makedirs(out_dir, exist_ok=True)
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print(f"  !! FAILED ({bl}, {p['tag']}) rc={rc}", flush=True)
            else:
                n_ok += 1
    print(f"=== PREDICT_TRIPLES {'DRY-RUN' if args.dry_run else 'DONE'} — {n_ok} run(s) ok ===", flush=True)


if __name__ == "__main__":
    main()
