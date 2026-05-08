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
Based on the [EDC framework](https://arxiv.org/abs/2404.03868), adapted for EMERGE in
the `edc-tt2kg/` directory **bundled in-tree** at `src/benchmarks/wrappers/edc_plus/edc_tt2kg/`
(no external clone needed — defaults to in-tree; set the `EDC_REPO_PATH` env var only
if you want to point at an alternate checkout).

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

The orchestrator (`run_benchmarks.py`) lives in the **`emerge`** conda env (per `requirements/core.txt`). It dispatches to a per-model wrapper that activates **its own** conda env (per `requirements/benchmarks/<model>.txt`). Both envs must exist before running.

> **SLURM users:** equivalent sbatch scripts in `scripts/slurm/benchmarks/`.

### Quick start: run EDC+ ZS GPT-5.1 from a fresh clone

End-to-end walkthrough on a fresh clone. **All commands run from the repository root.**

#### 1. Set up the two conda envs

```bash
# Orchestrator env (also used for ReLiK)
conda create -n emerge python=3.10 -y && conda activate emerge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements/core.txt

# EDC+ wrapper env (run.sh activates this inside the subprocess)
conda create -n edc python=3.9 -y && conda activate edc
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/benchmarks/edc-plus.txt
```

#### 2. Download the data EDC+ needs

```bash
conda activate emerge
./scripts/download_data.sh --indices    # ~155 MB eval set + ~400 MB indices
```

EDC+ reads `data/evaluation_set/` (default) **and** `data/indices/relik_edc_relation_indexes/` (only with `--indices`). Skip `--indices` and you'll hit `AssertionError` at `s01_run_v3.py:82` — this was [issue #1](https://github.com/klimzaporojets/emerge/issues/1).

#### 3. Set the API key

```bash
export OPENAI_API_KEY_MY='your-openai-key-here'
# Note: edc_plus_zs_gpt_5_1 reads OPENAI_API_KEY_MY; the ICL configs read OPENAI_API_KEY.
# See the per-config "Backend LLM" table above for which env var each variant uses.
```

#### 4. Run

```bash
conda activate emerge   # orchestrator env; the wrapper subprocess switches to `edc`
export PYTHONPATH="$PWD/src"

mkdir -p logs
python -u -m benchmarks.run_benchmarks \
    --config_file config/benchmarks/s02_run_benchmarks/edc_plus_zs_gpt_5_1/config.json \
    2>&1 | tee logs/edc_zs_gpt_5_1.log
```

#### 5. What success looks like

- Console: `args_value_is: {...}` (config loaded), per-delta `processing ...` lines
- Output: 7 snapshots × 5 deltas = 35 JSONL files under `output/s02_run_benchmarks/edc_plus_zs_gpt_5_1/last_processed/snapshot_*-01-01/`, each with 100 instances

**Common failures** (with fix):
- `AssertionError` at `s01_run_v3.py:82` → forgot `--indices` in step 2
- `aiohttp ClientResponseError: 401 Unauthorized` → API key env var unset or wrong name (check step 3)
- `ModuleNotFoundError: openai` → the `edc` conda env isn't activated in the subprocess; check that step 1's `pip install -r requirements/benchmarks/edc-plus.txt` succeeded

### Per-model differences (everything else)

Same command shape as the EDC+ walkthrough; only these fields change. Replace `<config>` with the directory name shown.

| Model | Conda env (wrapper) | API key env var | `download_data.sh` flag | GPU |
|---|---|---|---|---|
| **EDC+ ICL GPT-5.1** (`edc_plus_icl_gpt_5_1`) | `edc` | `OPENAI_API_KEY` | `--indices` | — |
| **EDC+ ICL Mistral L/S** (`edc_plus_icl_mistral_{large,small}`) | `edc` | `AZURE_AI_API_KEY` | `--indices` | — |
| **EDC+ ZS Mistral L** (`edc_plus_zs_mistral_large`) | `edc` | `AZURE_AI_API_KEY` | `--indices` | — |
| **KGGen GPT-5.1** (`20260116_kggen_gpt_5_1`) | `kggen-py312` | `AZURE_OPENAI_API_KEY` | (none beyond default) | — |
| **KGGen Mistral L/S** (`20260114_kggen_mistral_{large,small}`) | `kggen-py312` | `AZURE_AI_API_KEY` | (none beyond default) | — |
| **RAKG Mistral L/S** (`20260114_rakg_mistral_{large,small}`) | `rakg-py311` *plus* `git clone https://github.com/RUC-NLPIR/RAKG.git && export RAKG_REPO_PATH=$PWD/RAKG` | `AZURE_AI_API_KEY` | (none beyond default) | — |
| **REBEL** (`20260114_rebel`) | `rebel-py311` | — (local model) | (none beyond default) | 1 × 16 GB+ VRAM, 100 GB RAM |
| **ReLiK Open IE** (`relik_oie`) | `emerge` (same as orchestrator) | — | `--indices` | 1 × 16 GB+ VRAM, 64 GB RAM |
| **ReLiK Closed IE** (`relik_cie`) | `emerge` | — | `--indices` *and* `--kg` (~22 GB after decompression) | 1 × 16 GB+ VRAM, 180 GB RAM |

For each: `pip install -r requirements/benchmarks/<model>.txt` to populate the wrapper env (the file's header has the exact `conda create` command).

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
