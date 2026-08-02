# EMERGE: A Benchmark for Updating Knowledge Graphs with Emerging Textual Knowledge

**[Paper](https://arxiv.org/abs/2507.03617)** | **[Dataset](https://huggingface.co/datasets/klimzaporojets/emerge-benchmark)** | **[Code](https://github.com/klimzaporojets/emerge)**

Repository for the **EMERGE** benchmark: a dataset and evaluation framework for
**Text-driven KG Updating (TKGU)** — evaluating methods for updating knowledge
graphs from textual evidence.

> ### ⭐ KG-aware baseline
> A targeted **GPT-5.1 baseline with access to the KG snapshot** (`oracle` / `kg_rag@32` /
> `oracle+kg_rag`), added to test whether feeding KG structure closes the gap the IE-only baselines
> leave. **→ [Results table and how to reproduce it](#kg-aware-baseline)**
> _Added 2026-07-27 for the NeurIPS 2026 rebuttal._

> Klim Zaporojets, Daniel Daza, Edoardo Barba, Ira Assent, Roberto Navigli, Paul Groth

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/klimzaporojets/emerge.git
cd emerge
conda create -n emerge python=3.12 -y && conda activate emerge
pip install torch torchvision torchaudio
pip install -r requirements/core.txt

# 2. Download the dataset (~155 MB)
./scripts/download_data.sh
# Optional — for correct relik-cie scoring in step 3 (to identify Exists vs.
# Add TKGU operations), also download KG snapshots:
# ./scripts/download_data.sh --kg     # +22 GB after decompression

# 3. Reproduce paper evaluation (requires 1× GPU)
./scripts/run/evaluate.sh

# 4. View results in notebooks (saved outputs included — no re-execution needed)
jupyter lab src/stats/
```

Notes:
- **GPU users**: pin a CUDA-matching `torch` wheel (see https://pytorch.org/) BEFORE step 1's `core.txt` install — e.g. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` for CUDA 12.1.
- **Optional downloads** beyond the default test set:
  - `./scripts/download_data.sh --indices` (+400 MB) — relation indexes used by EDC+ and ReLiK
  - `./scripts/download_data.sh --corpus` (+2.4 GB) — 1.19M-instance corpus used by Part 3 garbage QA + paper §4.3 statistics
  - `./scripts/download_data.sh --kg` (+22 GB after decompression) — KG snapshots, only needed for correct `relik-cie` scoring (see [Reproducing paper results](#reproducing-paper-results))
- **Step 3 hardware + time**: see the [Reproducing paper results](#reproducing-paper-results) section for full details (RAM/GPU/runtime, plus what happens if KG snapshots are not present on disk).

---

Each EMERGE instance pairs a textual passage with a KG snapshot and a set of update
operations induced by the passage. EMERGE defines five TKGU operations:

| Operation | Description |
|-----------|-------------|
| **Exists** (E) | Triple already present in the KG, supported by the textual passage |
| **Add** (A) | New triple involving entities that already exist in the KG |
| **Mint+Add** (M+A) | New triple involving one or more entities not yet in the KG |
| **Infer** (I) | Triple linking a newly introduced entity to an existing KG entity, not explicitly stated in the passage |
| **Deprecate** (D) | Existing triple invalidated by updated information in the passage |

> **Note on internal naming:** The codebase uses different identifiers for TKGU operations:
> Exists = `x-triples`, Add = `e-triples`, Mint+Add = `ee-triples`,
> Infer = `ee-kg-triples`, Deprecate = `d-triples`.

## Dataset

The dataset is hosted on [HuggingFace Hub](https://huggingface.co/datasets/klimzaporojets/emerge-benchmark)
and must be downloaded before use (see [Setup](#setup)).

The released JSONL has been QA'd via an iterative re-query pipeline
documented in [`src/dataset/README.md` Part 3](src/dataset/README.md);
paper §4.3 / Table 8 / Datasheet K.2 statistics (608K Exists, 207K Add,
149K Mint+Add, 220K Infer, 9.5K Deprecate, 1.19M total) are reproducible
with `scripts/stats/compute_405bv1_dataset_stats.py` after running
`./scripts/download_data.sh --corpus`.

The test set contains 3,500 instances across 7 annual Wikidata snapshots (2019–2025).
Each instance contains:

- A Wikipedia passage with temporal context (snapshot date, delta dates)
- Ground-truth TKGU triples with LLM and human assessments
- Predictions from 13 benchmark models (plus a **KG-aware baseline** — see [KG-aware baseline](#kg-aware-baseline))

| Model (paper name) | Type | Backend |
|-------------------|------|---------|
| EDC+ GPT-5.1 | LLM-based (in-context learning) | GPT-5.1 |
| EDC+ M-Lg | LLM-based (in-context learning) | Mistral-Large |
| EDC+ M-Sm | LLM-based (in-context learning) | Mistral-Small |
| EDC+ ZS GPT-5.1 | LLM-based (zero-shot) | GPT-5.1 |
| EDC+ ZS M-Lg | LLM-based (zero-shot) | Mistral-Large |
| KGGen GPT-5.1 | LLM-based | GPT-5.1 |
| KGGen M-Lg | LLM-based | Mistral-Large |
| KGGen M-Sm | LLM-based | Mistral-Small |
| RAKG M-Lg | LLM-based | Mistral-Large |
| RAKG M-Sm | LLM-based | Mistral-Small |
| REBEL | Local seq2seq | Babelscape/rebel-large |
| ReLiK RE | Local neural (Open IE) | sapienzanlp/relik-relation-extraction-nyt-large |
| ReLiK cIE | Local neural (Closed IE) | sapienzanlp/relik-cie-large |
| **KG-aware** | LLM-based (KG-conditioned) | GPT-5.1 |

The first 13 rows are the released baselines whose predictions ship in `data/evaluation_set/`. The
last row, **KG-aware**, is a targeted baseline that gives GPT-5.1 access to the KG snapshot to test
whether feeding KG structure closes the gap the IE-only methods leave — see
[KG-aware baseline](#kg-aware-baseline) for its variants, results, and reproduction.

## Dataset construction pipeline

The released `data/evaluation_set/` is the **frozen final artifact** of the pipeline
below. **Using or evaluating against the dataset does NOT require running any of these
steps** — `download_data.sh` + `evaluate.sh` is sufficient. The construction code is
shipped here for **transparency**.

Each `s0x` label below is a pipeline-step identifier; code lives under `src/dataset/{wikidata,wikipedia,emerge}/<step>_*.py` and SLURM submitters in `scripts/slurm/{wikidata,wikipedia,dataset}/<step>_*.sh`. See [`src/dataset/README.md`](src/dataset/README.md) for per-step execution details + config field semantics.

```
Raw Wikidata + Wikipedia dumps  (~TB, fetch from dumps.wikimedia.org — not in repo)
                                                       │
                                                       ▼
                  [WD s01–s03 / WP s01–s03]
                  Wikidata triple history → KG snapshots + per-triple deltas
                  Wikipedia history → entity descriptions + hyperlinks + title changes
                                                       │
                                                       ▼
                  [emerge s03 / s04]
                  Match passages ↔ KG-change deltas, filter for KG-relevant snippets
                                                       │
                                                       ▼
                  ┌────────────────────────────────────────────────────────────────────┐
                  │  [s05]   LLM annotation (Llama 8B / 405B w/ TGI, prompts v1)        │
                  │    ↑                                                       ↓        │
                  │    └──── iterative annotation refinement of likely-incorrect ◀┘     │
                  │          annotations (`find_garbage_clusters.py`)                   │
                  └────────────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
                  [s07]   Global dedup using `llm_assessment` ranking
                          (Jaccard ≤ 0.50, edit-dist ≥ 0.20)
                                                       │
                                                       ├──────►  data/corpus/   (released)
                                                       │
                                                       ▼
                  [s07b]  Build the 3,500-instance test set: subsample (100 per
                          delta, ≥ 40 per d/ee_kg/e/ee operation type), reformat
                          to final schema, verify
                                                       │
                                                       ▼
                  [s09]   Human annotation (subset, for inter-annotator agreement)
                                                       │
                                                       ▼
                                          data/evaluation_set/   (released, 3,500)
```

### What's in this repo for each step

**All pipeline code is shipped.** It is built for iterative LLM annotation: s05's output schema
equals its input schema, so you can re-run s05 on either released artifact — `data/evaluation_set/`
(3,500) or `data/corpus/` (1.19M, via `--corpus`) — with any LLM you can serve. Both already carry
the paper's `Meta-Llama-3.1-405B-Instruct-GPTQ-INT4` assessments, so re-running s05 is only needed to
benchmark a different LLM or re-query triples (as Part 3 does).

| Step | Shipped | Run on released data? |
|---|---|---|
| **s05** LLM assessment (Llama 8B / 405B, prompts v1) | runner + SLURM (`_llama8b`/`_llama405b`) + TGI restart loop + prompts + runnable config † | ✅ re-annotate eval_set / corpus (1× / 4× H100) |
| **Part 3** QC (detect → reinput → merge) | `scripts/dataset/` (4 CLI tools) | ✅ on `data/corpus/` — reproduces 220 flagged (0.0113 %) |
| **§4.3 / Table 8** stats | `scripts/stats/` (3 CLI tools) | ✅ on `data/corpus/` — reproduces 608K/207K/149K/220K/9.5K/1.19M |
| **s09** human agreement | `src/dataset/emerge/s09{a–e}_*.py` | ✅ `annotation_agreement.sh` (new annotations need annotators) |
| **WD/WP s01–s03, emerge s03/s04** | code + SLURM submitters | placeholder configs (need raw dumps) |
| **s06b → s07 → s07b → s07c** (reformat → dedup → subsample) | `src/dataset/emerge/{s06b,s07,s07b,s07c}_*.py` | placeholder configs (depend on s05 outputs) |
| **evaluate.sh** paper results | `scripts/run/evaluate.sh` | ✅ reproduces the main results table |

† Runnable s05 config: `config/dataset/emerge/s05_generate_dataset_with_llm/20260510_s05_runnable_on_released/` (points at `./data/evaluation_set/`). You supply the TGI Apptainer container, Llama weights (auto-pulled), and 3 env paths in the SLURM script.

### Not yet shipped

The raw Wikidata + Wikipedia history dumps (~TB) and their s01–s03 / s03–s04 intermediates are too
large to redistribute — fetch the dumps from [dumps.wikimedia.org](https://dumps.wikimedia.org/)
(`wikidatawiki-*-pages-meta-history*.7z`, `enwiki-*-pages-meta-history*.7z`); re-creating
`data/corpus/` from scratch means running WD/WP/s03/s04 yourself. We plan to ship more intermediates
(and a dump-download helper) in future releases.

For per-step details + config field semantics, see [`src/dataset/README.md`](src/dataset/README.md).

## Project structure

```
src/
├── dataset/                    Dataset creation pipeline
│   ├── emerge/                 EMERGE dataset construction (s03–s09)
│   │   ├── utils/              Shared utilities (constants, LLM prompts, text processing)
│   │   └── prompts/            LLM prompt templates for triple assessment
│   ├── wikidata/               Wikidata KG extraction (Java + Python)
│   └── wikipedia/              Wikipedia processing (hyperlinks, entity descriptions)
│
├── evaluation/                 Model evaluation pipeline
│   ├── s0x_evaluate_predictions.py    Main entry point
│   ├── evaluator/              Metric implementations (completeness, graph_judge, CIE)
│   ├── preloader/              Data loading + preprocessing
│   └── scorers/                Scoring (BERTScore, sentence transformers, exact match)
│
├── benchmarks/                 Benchmark model wrappers (13 models)
│   ├── run_benchmarks.py       Orchestrator: dispatches to model wrappers
│   ├── wrappers/               Per-model execution: kggen/, rakg/, rebel/, edc_plus/, relik/
│   └── configs/                Model-specific configuration classes
│
├── merge/                      Merge predictions + LLM assessments + human annotations
│
└── stats/                      Paper statistics (Jupyter notebooks)
    ├── evaluation/             Model performance tables and figures
    └── dataset/                Dataset statistics and annotation agreement

requirements/
├── core.txt                    Evaluation + statistics + merge
├── dataset.txt                 Dataset creation pipeline (extends core)
└── benchmarks/                 Per-model requirements (separate conda envs)
    ├── edc-plus.txt            EDC+ (Python 3.9)
    ├── kggen.txt               KG-GEN (Python 3.12)
    ├── rakg.txt                RAKG (Python 3.11)
    ├── rebel.txt               REBEL (Python 3.11)
    └── relik.txt               ReLiK (shares core env)

scripts/
├── download_data.sh            Download dataset from HuggingFace Hub (--kg, --indices, --corpus, --all)
├── dataset/                    Garbage detect → reinput → merge chain (4 CLI tools, see src/dataset/README.md Part 3)
├── stats/                      Paper §4.3 / Table 8 dataset-stat reproducibility (3 CLI tools, run on data/corpus/)
├── run/                        Non-SLURM entry points (evaluate, benchmarks, etc.)
├── slurm/                      Optional SLURM sbatch scripts (for HPC clusters)
├── test_dataset_stats.sh       Lightweight smoke test for the dataset/stats path (CPU only, ~5 s)
└── test_end_to_end.sh          Heavy benchmark-evaluation test (GPU + 180 GB RAM, designed for an HPC node)

tests/                          Pytest suite for the dataset construction pipeline (CPU only, ~3 s)
├── conftest.py
├── test_s05_*.py               s05 prompt-builder regressions (mock + AST + signature audit)
├── test_dataset_integrity.py   Released-data sanity checks (record count, hash uniqueness, schema)
├── test_config_files.py        JSON validity + path hygiene on every config/**/*.json
├── test_chain_scripts.py       Synthetic-tmp_path unit tests for the garbage-detect chain
└── test_ported_scripts_argparse.py    Smoke test on each scripts/dataset/ + scripts/stats/ tool
```

## Requirements

- **Python 3.12** with conda
- **GPU**: 1x GPU with 16GB+ VRAM for evaluation (BERTScore, sentence-transformers)
- **RAM**: 32GB minimum. 180GB+ only if loading full KG snapshots for Exists operation evaluation of CIE models
- **API keys** (only for LLM-based benchmark models): Azure OpenAI and/or OpenAI

## Setup

The [Quick Start](#quick-start) covers the base install (`emerge` env + `core.txt`) and the default
dataset download. This section adds the details.

After installing `core.txt`, download the NLTK data:
```bash
python -c "import nltk; nltk.download('words'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

> **Benchmark models** need their own conda environments (package conflicts) — see `requirements/benchmarks/`.
> **Dataset creation** needs `pip install -r requirements/dataset.txt`.

`./scripts/download_data.sh` populates `data/`:

```
data/
├── evaluation_set/    Test set: 7 snapshots × 5 deltas × 100 = 3,500 instances (+ 13 model predictions)
│                      snapshot_{2019..2025}-01-01/delta_*.jsonl
└── human_annotation/  solved_disagreements.jsonl   (for agreement stats)
```

Optional downloads: `--kg` (~22 GB KG snapshots, needed only to score `relik-cie` — without them
`evaluate.sh` skips relik-cie and prints a NOTE), `--indices` (relation indexes), `--corpus`
(1.19M-instance corpus).

## Reproducing paper results

All commands assume you are in the repository root with `conda activate emerge`.
They work on **any machine** — no SLURM or HPC cluster required.

> **SLURM users:** Equivalent sbatch scripts are in `scripts/slurm/`. Each script
> sets up the environment and calls the same Python commands shown below.

### Evaluate model predictions

The evaluation pipeline scores the 13 benchmark model predictions (included in the
dataset) against ground-truth annotations:

```bash
./scripts/run/evaluate.sh
```

This uses the recommended config (all 13 models, correct scoring, with KG snapshots)
and produces a `wiki_eval_result.pkl` file that the statistics notebooks read.

To use a different config or run manually:
```bash
export PYTHONPATH="$PWD/src"
python -u -m evaluation.s0x_evaluate_predictions \
    --config_file config/evaluation/s0x_evaluate_predictions/20260324_all_models_with_zs_fixed_with_kg/config.json
```

See [`src/evaluation/README.md`](src/evaluation/README.md) for all available configs
and the difference between **fixed** vs **legacy** scoring modes.

### Reproduce paper tables and figures

All paper tables and figures are generated from Jupyter notebooks in `src/stats/`.
The notebooks include **saved cell outputs**, so you can view all results (tables,
figures, numbers) directly **without re-executing**.

To re-execute them yourself (requires the evaluation pkl from the step above):
```bash
jupyter lab --no-browser --port=8888
# Open http://localhost:8888 in your browser
# Navigate to src/stats/evaluation/ or src/stats/dataset/
```

| Notebook | Paper content |
|----------|--------------|
| `src/stats/evaluation/tables_results.ipynb` | Main results table (Completeness + G-BERTScore-R) |
| `src/stats/evaluation/figure_zs_vs_icl.ipynb` | Zero-shot vs in-context learning figure |
| `src/stats/evaluation/figure_deltas_snapshots.ipynb` | Score vs delta/snapshot plots |
| `src/stats/evaluation/table_cie_exact_match.ipynb` | QID exact-match results |
| `src/stats/dataset/figure_tkgu_distribution.ipynb` | TKGU operation distribution |
| `src/stats/dataset/appendix_table_annotation_agreement.ipynb` | Annotation agreement table |

See [`src/stats/README.md`](src/stats/README.md) for the complete notebook inventory.

### Reproduce annotation agreement

Computes inter-annotator agreement (Cohen's kappa, Fleiss' kappa, Krippendorff's alpha)
between two human annotators and the LLM assessor. No GPU needed:

```bash
./scripts/run/annotation_agreement.sh
```

Output includes a LaTeX table matching the annotation agreement table in the paper.

## Running benchmark models

To re-run any of the 13 benchmark models on the dataset:

```bash
# LLM-based models (KG-GEN, RAKG, EDC+) — require API keys, CPU only:
export AZURE_OPENAI_API_KEY='your-key-here'
./scripts/run/run_benchmark.sh config/benchmarks/s02_run_benchmarks/20260116_kggen_gpt_5_1/config.json

# Local models (REBEL, ReLiK) — require GPU, no API keys:
./scripts/run/run_benchmark.sh config/benchmarks/s02_run_benchmarks/20260114_rebel/config.json
```

Or equivalently, without the wrapper script:

```bash
export PYTHONPATH="$PWD/src"
export AZURE_OPENAI_API_KEY='your-key-here'

python -u -m benchmarks.run_benchmarks \
    --config_file config/benchmarks/s02_run_benchmarks/20260116_kggen_gpt_5_1/config.json
```

Each benchmark model requires its own conda environment due to package conflicts.
Run `./scripts/run/run_benchmark.sh --help` for the full list of available configs
and required environments.
See [`src/benchmarks/README.md`](src/benchmarks/README.md) for detailed per-model instructions.

> **Heads-up on `relik_cie`**: re-running ReLiK Closed IE additionally requires a
> per-snapshot Wikipedia entity index that is **not** part of `download_data.sh`.
> Building it via `scripts/slurm/dataset/s08_extract_relik_dictionary.sh` needs
> ~80 CPUs, 128 GB RAM, and up to 48 h of wall-clock against an English Wikipedia
> history dump. The released `data/evaluation_set/` already contains the
> `relik-cie` predictions, so evaluating ReLiK Closed IE against the paper numbers
> works without this index — re-generation only. See
> [`src/benchmarks/README.md`](src/benchmarks/README.md#note-relik-closed-ie-entity-index-heavy)
> for the full breakdown.

## KG-aware baseline

_Added 2026-07-27 for the NeurIPS 2026 rebuttal._

A targeted baseline that gives the **same GPT-5.1 used by EDC+** access to the reference KG
snapshot `G_t`, keeping the model fixed and varying only its access to KG structure. It tests
directly whether feeding KG context closes the gap that IE-only baselines leave. Three variants:
`oracle` (gold entities/relation types and existing triples), `kg_rag@32` (top-32 candidates
retrieved from `G_t` with ReLiK), and `oracle+kg_rag` (oracle context plus retrieved distractors).

Scored with the same metrics as Table 3 — Completeness (C), G-BERTScore-R (G-R), Executable-R (E-R)
— on the full 3,500-instance test set. Best per operation per metric in **bold**; `–` = not
computable (open-IE methods produce no canonical IDs; MINT+ADD / INFER mint entities with no ID
in `G_t`); `—` = operation not supported by that method.

| Method | Exists (C / G-R / E-R) | Add (C / G-R / E-R) | Mint+Add (C / G-R / E-R) | Infer (C / G-R / E-R) | Deprecate (C / G-R / E-R) |
|---|---|---|---|---|---|
| KG-aware (oracle) | **60.4** / 50.3 / **50.3** | **46.5** / 42.9 / **39.6** | **43.2** / 47.6 / – | **50.3** / **75.9** / – | **38.4** / 36.9 / **36.9** |
| KG-aware (kg_rag@32) | 13.8 / 27.5 / 19.4 | 15.3 / 18.7 / 12.5 | 30.6 / 49.7 / – | 11.3 / 52.1 / – | 12.3 / 12.7 / 11.2 |
| KG-aware (oracle+kg_rag) | 52.6 / 49.3 / 43.2 | 27.4 / 27.0 / 18.2 | 39.3 / 53.1 / – | 45.0 / 75.5 / – | 18.7 / 18.4 / 17.3 |
| EDC+ GPT-5.1 (no KG) | 25.7 / **66.3** / – | 38.0 / **76.6** / – | 35.5 / **78.2** / – | 24.3 / 73.2 / – | 36.5 / **54.8** / – |
| ReLiK cIE | 15.2 / 30.7 / 15.0 | 18.1 / 57.5 / 13.2 | — | — | — |

With `oracle` context, the same GPT-5.1 gives the strongest near-exact updates (C) on every operation
and — by canonicalizing against `G_t` — uniquely produces executable updates (E-R) where the
entities already exist in the KG (EXISTS, ADD, DEPRECATE); DEPRECATE is executed by no other baseline.
The gap between `oracle` and `kg_rag@32` shows KG retrieval is a major bottleneck, and `oracle` not
saturating shows a residual reasoning gap even with correct context. EDC+'s higher G-R reflects that
metric rewarding semantic similarity rather than the exact KG form.

**To reproduce these numbers**, see [`src/kg_context/README.md`](src/kg_context/README.md) — the
AUGMENT → COMPOSE → PREDICT → SCORE runbook (configs, conda envs, commands, and a cheap smoke test).

## Evaluating new benchmark predictions

Wrappers write their predictions under `./output/s02_run_benchmarks/<config_dir>/`
with model-local names (e.g., the KGGen wrapper writes `kg-gen-gpt-5.1`, while the
canonical key the evaluator expects is `kg-gen/azure/gpt5.1`). Before evaluation,
the **merge step** (`src/merge/s0x_merge_predictions.py`) reads each wrapper's raw
output, normalises the model name to its canonical form, and writes a unified
dataset that the evaluator consumes. Pipeline:

```
wrapper → merge → evaluate
```

Skipping merge → the evaluator silently filters out non-canonical names and
reports **0 results** for your re-run model.

```bash
conda activate emerge
export PYTHONPATH="$PWD/src"

# 1. Re-run one or more wrappers (see "Running benchmark models" above).

# 2. Merge re-run predictions into the released main dataset.
#    The config has a `_help` field at the top documenting the field semantics.
#    Delete entries from `input_other_datasets_paths` for any model you did NOT
#    re-run — merge raises FileNotFoundError on missing wrapper output paths.
python -u -m merge.s0x_merge_predictions \
    --config_file config/merge/s0x_merge_predictions/20260509_all_models_with_zs/config.json
# → writes ./output/s0x_merge_predictions/20260509_all_models_with_zs/

# 3. Evaluate. Copy a config under config/evaluation/s0x_evaluate_predictions/
#    and change `input_dataset_path` to the merge output above before running.
./scripts/run/evaluate.sh path/to/your/evaluation_config.json
```

Reproducing the released paper numbers does **not** require the merge step —
`./scripts/run/evaluate.sh` reads the released `data/evaluation_set/` directly,
which already contains predictions from all 13 baseline models. Merge is only
needed when you want to evaluate **new** predictions you re-ran.

## Two scoring modes

| Mode | `score_empty_predictions_as_zero` | Description |
|------|----------------------------------|-------------|
| **Fixed** (correct) | `true` | Instances with no predictions scored as P=0/R=0/F1=0 |
| **Legacy** | `false` | Instances with no predictions excluded from averages (reproduces original submission) |

## Tests

A lightweight pytest suite in `tests/` exercises the dataset construction
pipeline (`src/dataset/emerge/`) and the chain CLI tools in
`scripts/dataset/` and `scripts/stats/`. CPU only, no GPU, no network, no
LLM API. Runs in ~3 s; data-dependent tests skip cleanly on a fresh clone.

```bash
pip install pytest                # one-time
pytest tests/
```

A complementary shell smoke runner bundles the same checks plus a
no-data-required argparse smoke and an optional paper-numbers
reproducibility pass:

```bash
bash scripts/test_dataset_stats.sh             # ~5 s, no data needed
bash scripts/test_dataset_stats.sh --with-data # +1 min: pulls 2.3 GB corpus and asserts
                                               # paper §4.3 / Table 8 numbers reproduce
```

The heavier `bash scripts/test_end_to_end.sh` runs the **full benchmark
evaluation** (requires GPU + 180 GB RAM, designed for an HPC node) and
is not part of the routine test suite.

## Key documentation

| File | What it covers |
|------|---------------|
| [`src/evaluation/README.md`](src/evaluation/README.md) | Evaluation pipeline: metrics, caching, scoring modes (incl. `cie_exact_match` ↔ Executable-R / E-R mapping) |
| [`src/benchmarks/README.md`](src/benchmarks/README.md) | All 13 benchmark models: architectures, APIs, configs |
| [`src/kg_context/README.md`](src/kg_context/README.md) | KG-aware baseline (GPT-5.1 + KG context): AUGMENT → COMPOSE → PREDICT → SCORE runbook, configs, envs, results |
| [`src/dataset/README.md`](src/dataset/README.md) | Full dataset creation pipeline (Wikidata + Wikipedia + EMERGE) — including **Part 3: Quality control of LLM annotations** (the detect → reinput → merge workflow exposed by `scripts/dataset/`) |
| [`src/stats/README.md`](src/stats/README.md) | Paper statistics: notebook inventory (model evaluation), data dependencies. **Note**: `src/stats/` is notebooks for evaluation tables/figures; the CLI scripts in `scripts/stats/` are for **dataset** stat reproducibility (different artifacts despite the shared name). |

## Citation

```bibtex
@article{zaporojets2025emerge,
  title={EMERGE: A Benchmark for Updating Knowledge Graphs with Emerging Textual Knowledge},
  author={Zaporojets, Klim and Daza, Daniel and Barba, Edoardo and Assent, Ira and Navigli, Roberto and Groth, Paul},
  journal={arXiv preprint arXiv:2507.03617},
  year={2025}
}
```

## License

- **Code:** [Apache License 2.0](LICENSE)
- **Dataset:** [Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)](LICENSE-DATA)

The dataset is derived from [Wikipedia](https://en.wikipedia.org/) (CC BY-SA 3.0+) and [Wikidata](https://www.wikidata.org/) (CC0).
