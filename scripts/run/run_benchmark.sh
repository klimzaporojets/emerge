#!/bin/bash
# Run a benchmark model on the EMERGE dataset.
#
# This script invokes the benchmark orchestrator, which dispatches to the
# correct model wrapper based on the config file. Each wrapper activates
# its own conda environment internally.
#
# Requirements:
#   - conda env: emerge (see requirements/core.txt) — for the orchestrator
#   - Model-specific conda env (see below) — activated by the wrapper
#   - API keys (for LLM-based models): set as environment variables before running
#
# Conda environments needed per model:
#   emerge          — ReLiK (relik_oie, relik_cie)
#   edc             — EDC+ (edc_plus_icl_*, edc_plus_zs_*)
#   kggen-py312     — KG-GEN (20260114_kggen_*, 20260116_kggen_*)
#   rakg-py311      — RAKG (20260114_rakg_*)
#   rebel-py311     — REBEL (20260114_rebel)
#
# Resource requirements per model:
#   KG-GEN, RAKG, EDC+  — CPU only, 8-32GB RAM (API-based)
#   REBEL                — 1x GPU (16GB+ VRAM), 32GB RAM
#   ReLiK                — 1x GPU (16GB+ VRAM), 64GB RAM
#
# Usage:
#   ./scripts/run/run_benchmark.sh <config_file>
#
# Examples:
#   # KG-GEN with GPT-5.1 (needs AZURE_OPENAI_API_KEY):
#   export AZURE_OPENAI_API_KEY='your-key'
#   ./scripts/run/run_benchmark.sh config/benchmarks/s02_run_benchmarks/20260116_kggen_gpt_5_1/config.json
#
#   # REBEL (local GPU, no API key needed):
#   ./scripts/run/run_benchmark.sh config/benchmarks/s02_run_benchmarks/20260114_rebel/config.json
#
#   # EDC+ ICL with Mistral-Large (needs AZURE_AI_API_KEY):
#   export AZURE_AI_API_KEY='your-key'
#   ./scripts/run/run_benchmark.sh config/benchmarks/s02_run_benchmarks/edc_plus_icl_mistral_large/config.json
#
# Available configs:
#   config/benchmarks/s02_run_benchmarks/
#   ├── 20260116_kggen_gpt_5_1/          KG-GEN GPT-5.1
#   ├── 20260114_kggen_mistral_large/     KG-GEN Mistral-Large
#   ├── 20260114_kggen_mistral_small/     KG-GEN Mistral-Small
#   ├── 20260114_rakg_mistral_large/      RAKG Mistral-Large
#   ├── 20260114_rakg_mistral_small/      RAKG Mistral-Small
#   ├── 20260114_rebel/                   REBEL
#   ├── edc_plus_icl_gpt_5_1/            EDC+ ICL GPT-5.1
#   ├── edc_plus_icl_mistral_large/       EDC+ ICL Mistral-Large
#   ├── edc_plus_icl_mistral_small/       EDC+ ICL Mistral-Small
#   ├── edc_plus_zs_gpt_5_1/             EDC+ Zero-shot GPT-5.1
#   ├── edc_plus_zs_mistral_large/        EDC+ Zero-shot Mistral-Large
#   ├── relik_oie/                        ReLiK Open IE
#   └── relik_cie/                        ReLiK Closed IE

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file>"
    echo ""
    echo "Run '$0 --help' or see the script header for available configs."
    exit 1
fi

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    head -60 "$0" | grep "^#" | sed 's/^# \?//'
    exit 0
fi

CONFIG_FILE="$1"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -d "data/evaluation_set" ]; then
    echo "Error: dataset not found. Run ./scripts/download_data.sh first."
    exit 1
fi

echo "Running benchmark model"
echo "  Config: $CONFIG_FILE"
echo ""

mkdir -p logs

export PYTHONPATH="$PWD/src"
python -u -m benchmarks.run_benchmarks \
    --config_file "$CONFIG_FILE" \
    2>&1 | tee logs/benchmark_$(date +%Y%m%d_%H%M%S).log
