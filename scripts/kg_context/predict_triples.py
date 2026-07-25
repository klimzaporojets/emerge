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
import json
import os
import subprocess
import sys

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
    args = ap.parse_args()

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
