# Benchmark Models

This module runs 13 knowledge extraction models on the EMERGE dataset and produces
predictions in a format the evaluation pipeline can score.

> **Note on internal naming:** The codebase uses different identifiers for TKGU operations:
> Exists = `x-triples`, Add = `e-triples`, Mint+Add = `ee-triples`,
> Infer = `ee-kg-triples`, Deprecate = `d-triples`.

## Architecture

```
src/benchmarks/
├── run_benchmarks.py       Orchestrator: loads config, dispatches to the right wrapper
├── model_runner.py         Manages wrapper subprocess execution (run.sh + wrapper.py)
├── configs/                Model-specific config classes (experiment.py, kggen.py, rakg.py, rebel.py)
└── wrappers/
    ├── general_io.py       Shared I/O: reads dataset instances, writes predictions
    ├── prediction.py       Prediction data class
    ├── kggen/              KG-GEN wrapper (LLM API, ThreadPoolExecutor, cache with SHA256)
    │   ├── wrapper.py
    │   └── run.sh
    ├── rakg/               RAKG wrapper (external RAKG repo, retry logic for rate limits)
    │   ├── wrapper.py
    │   └── run.sh
    ├── rebel/              REBEL wrapper (HuggingFace transformers, 256-token chunks, GPU)
    │   ├── wrapper.py
    │   └── run.sh
    ├── edc_plus/           EDC+ wrapper (bridges to external edc-tt2kg repo)
    │   ├── wrapper.py
    │   └── run.sh
    └── relik/              ReLiK wrapper (in-process, per-snapshot entity/relation indices)
        ├── wrapper.py
        └── run.sh
```

## Model inventory

### LLM-based models (API calls, CPU-only on HPC cluster)

These models call external LLM APIs (Azure AI, Azure OpenAI, OpenAI) and run on CPU nodes.

#### EDC+ (Extract, Define, Canonicalize)

Three-stage LLM pipeline: Open IE extraction, schema definition, schema canonicalization.
Based on the [EDC framework](https://arxiv.org/abs/2404.03868), adapted for EMERGE in the
`edc-tt2kg` repository (external, referenced via absolute path in `run.sh`).

| Paper name | Internal name | Backend LLM | Setting | API |
|-----------|---------------|-------------|---------|-----|
| EDC+ GPT-5.1 | `edc-plus-open-ai/gpt-5.1/non-canonicalized` | GPT-5.1 | In-context learning (ICL) | OpenAI |
| EDC+ M-Lg | `edc-plus-azure_ai/Mistral-Large-2411` | Mistral Large | ICL | Azure AI |
| EDC+ M-Sm | `edc-plus-azure_ai/Mistral-small` | Mistral Small | ICL | Azure AI |
| EDC+ ZS GPT-5.1 | `edc-plus-zshot-open_ai/GPT-5_1` | GPT-5.1 | Zero-shot | OpenAI |
| EDC+ ZS M-Lg | `edc-plus-zshot-azure_ai/Mistral-Large-2411` | Mistral Large | Zero-shot | Azure AI |

- **ICL** models receive in-context examples from the dataset to guide extraction
- **Zero-shot** models receive only the task description, no examples
- The `non-canonicalized` variant outputs free-text triples (not linked to Wikidata QIDs)

#### KG-GEN (Knowledge Graph Generation)

LLM-based triple extraction using a KG-generation prompt strategy.
Uses the [KG-GEN pip package](https://pypi.org/project/kg-gen/).

| Paper name | Internal name | Backend LLM | API |
|-----------|---------------|-------------|-----|
| KGGen GPT-5.1 | `kg-gen/azure/gpt5.1` | GPT-5.1 | Azure OpenAI |
| KGGen M-Lg | `kg-gen/azure_ai/Mistral-Large-2411` | Mistral Large | Azure AI |
| KGGen M-Sm | `kg-gen/azure_ai/Mistral-small` | Mistral Small | Azure AI |

- Does **not** produce Deprecate operations (Deprecate operations show `--` in tables)

#### RAKG (Retrieval-Augmented Knowledge Graph)

LLM-based triple extraction with retrieval augmentation.
Uses the external [RAKG repository](https://github.com/RUC-NLPIR/RAKG) (referenced via absolute path in `run.sh`).

| Paper name | Internal name | Backend LLM | API |
|-----------|---------------|-------------|-----|
| RAKG M-Lg | `rakg/azure_ai/Mistral-Large-2411` | Mistral Large | Azure AI |
| RAKG M-Sm | `rakg/azure_ai/Mistral-small` | Mistral Small | Azure AI |

- Does **not** produce Deprecate operations (Deprecate operations show `--` in tables)

### Local models (GPU required)

These models run locally on GPU and do not call external APIs.

#### REBEL (Relation Extraction By End-to-end Language generation)

Seq2seq model that generates triples token-by-token. Splits input text into 256-token
chunks and processes them sequentially on a single GPU.

| Paper name | Internal name | HuggingFace model | GPU |
|-----------|---------------|-------------------|-----|
| REBEL | `rebel` | `Babelscape/rebel-large` | 1 GPU (100GB RAM) |

- Produces Exists and Add operations only
- Does **not** produce Mint+Add, Infer, or Deprecate operations (`--`)

#### ReLiK (Retriever, Linker, Knowledge)

Neural entity linking + relation extraction model. Runs in-process (not via wrapper subprocess).
Loads per-snapshot entity and relation indices for closed-IE evaluation.

| Paper name | Internal name | Type | GPU |
|-----------|---------------|------|-----|
| ReLiK RE | `relik-oie` | Open IE | 1 GPU (64GB RAM) |
| ReLiK cIE | `relik-cie` | Closed IE | 1 GPU (64GB RAM) |

- **Open IE**: produces free-text triples, no QID linking
- **Closed IE**: produces QID-linked triples. Requires KG snapshots loaded in memory
  during evaluation for correct Exists (x-triples) scoring
- Does **not** produce Deprecate operations (`--`)
- Closed IE does **not** produce Mint+Add or Infer operations (`--`)

## How to run

All commands work on **any machine** with the `emerge` conda environment activated.
Run from the repository root.

> **SLURM users:** Equivalent sbatch scripts are in `scripts/slurm/benchmarks/`.

### LLM API models (KG-GEN, RAKG) — CPU only

These models call external LLM APIs and do not need a GPU.

```bash
conda activate emerge
export PYTHONPATH="$PWD/src"

# Set the required API key (see API key table below)
export AZURE_OPENAI_API_KEY='your-key-here'

python -u -m benchmarks.run_benchmarks \
    --config_file config/benchmarks/s02_run_benchmarks/<model_config>/config.json \
    2>&1 | tee logs/<model_name>.log
```

Available configs:
| Config directory | Model | API key needed |
|-----------------|-------|----------------|
| `20260116_kggen_gpt_5_1/` | KG-GEN GPT 5.1 | `AZURE_OPENAI_API_KEY` |
| `20260114_kggen_mistral_large/` | KG-GEN Mistral Large | `AZURE_AI_API_KEY` |
| `20260114_kggen_mistral_small/` | KG-GEN Mistral Small | `AZURE_AI_API_KEY` |
| `20260114_rakg_mistral_large/` | RAKG Mistral Large | `AZURE_AI_API_KEY` |
| `20260114_rakg_mistral_small/` | RAKG Mistral Small | `AZURE_AI_API_KEY` |

### REBEL — requires GPU

```bash
conda activate emerge
export PYTHONPATH="$PWD/src"

python -u -m benchmarks.run_benchmarks \
    --config_file config/benchmarks/s02_run_benchmarks/20260114_rebel/config.json \
    2>&1 | tee logs/rebel.log
```

Requirements: 1 GPU, ~100GB RAM.

### EDC+ — CPU only (API calls)

EDC+ requires the external [EDC repository](https://github.com/bowen-zhang1/EDC)
cloned locally. Update the path in `src/benchmarks/wrappers/edc_plus/run.sh`.

```bash
conda activate emerge
export PYTHONPATH="$PWD/src"
export OPENAI_API_KEY='your-key-here'  # or AZURE_AI_API_KEY for Mistral models

python -u -m benchmarks.run_benchmarks \
    --config_file config/benchmarks/s02_run_benchmarks/<edc_config>/config.json \
    2>&1 | tee logs/<edc_model>.log
```

Available configs:
| Config directory | Model | API key needed |
|-----------------|-------|----------------|
| `edc_plus_icl_gpt_5_1/` | EDC+ ICL GPT 5.1 | `OPENAI_API_KEY` |
| `edc_plus_icl_mistral_large/` | EDC+ ICL Mistral Large | `AZURE_AI_API_KEY` |
| `edc_plus_icl_mistral_small/` | EDC+ ICL Mistral Small | `AZURE_AI_API_KEY` |
| `edc_plus_zs_gpt_5_1/` | EDC+ Zero-shot GPT 5.1 | `OPENAI_API_KEY` |
| `edc_plus_zs_mistral_large/` | EDC+ Zero-shot Mistral Large | `AZURE_AI_API_KEY` |

### ReLiK — requires GPU

```bash
conda activate emerge
export PYTHONPATH="$PWD/src"

python -u -m benchmarks.run_benchmarks \
    --config_file config/benchmarks/s02_run_benchmarks/<relik_config>/config.json \
    2>&1 | tee logs/<relik_model>.log
```

Available configs: `relik_oie/`, `relik_cie/`

Requirements: 1 GPU, ~64GB RAM. ReLiK cIE requires per-snapshot entity/relation indices.

## API key configuration

LLM API models require API keys set as environment variables before running:

| Model | Environment variable | API |
|-------|---------------------|-----|
| KG-GEN (GPT-5.1) | `AZURE_OPENAI_API_KEY` | Azure OpenAI |
| KG-GEN (Mistral) | `AZURE_AI_API_KEY` | Azure AI |
| RAKG (Mistral) | `AZURE_AI_API_KEY` | Azure AI |
| EDC+ (GPT-5.1) | `OPENAI_API_KEY` | OpenAI |
| EDC+ (Mistral) | `AZURE_AI_API_KEY` | Azure AI |

Set these in your shell before running sbatch, or add them to the sbatch script.

## Output format

All models produce per-instance predictions that are written to the output directory
specified in each config. These outputs are then combined by the merge step
(`src/merge/s0x_merge_predictions.py`) into the unified dataset format that the
evaluation pipeline reads.

## Logging

Log level is configurable via the `LOGGING_LEVEL` environment variable (default: `INFO`).
Set `LOGGING_LEVEL=DEBUG` for verbose output.

## References

- EDC+ is based on the [EDC framework](https://arxiv.org/abs/2404.03868)
- RAKG uses the [RAKG repository](https://github.com/RUC-NLPIR/RAKG)
- KG-GEN uses the [kg-gen pip package](https://pypi.org/project/kg-gen/)
- REBEL uses [Babelscape/rebel-large](https://huggingface.co/Babelscape/rebel-large)
- ReLiK uses [sapienzanlp/relik-relation-extraction-nyt-large](https://huggingface.co/sapienzanlp/relik-relation-extraction-nyt-large)
