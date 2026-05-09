# EMERGE: A Benchmark for Updating Knowledge Graphs with Emerging Textual Knowledge

**[Paper](https://arxiv.org/abs/2507.03617)** | **[Dataset](https://huggingface.co/datasets/klimzaporojets/emerge-benchmark)** | **[Code](https://github.com/klimzaporojets/emerge)**

Repository for the **EMERGE** benchmark: a dataset and evaluation framework for
**Text-driven KG Updating (TKGU)** — evaluating methods for updating knowledge
graphs from textual evidence.

> Klim Zaporojets, Daniel Daza, Edoardo Barba, Ira Assent, Roberto Navigli, Paul Groth

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/klimzaporojets/emerge.git
cd emerge
conda create -n emerge python=3.12 -y && conda activate emerge
# PyTorch — default install (CPU on most systems). For GPU, pin the CUDA-matching
# build first per https://pytorch.org/ (e.g. `pip install torch torchvision torchaudio
# --index-url https://download.pytorch.org/whl/cu121` for CUDA 12.1) BEFORE the
# core.txt install below.
pip install torch torchvision torchaudio
pip install -r requirements/core.txt

# 2. Download the dataset (test set + human annotations, ~155 MB)
./scripts/download_data.sh
# For benchmark runs, also fetch what each model needs:
#   ./scripts/download_data.sh --indices   # +400 MB — required by EDC+, ReLiK
#   ./scripts/download_data.sh --kg        # +3.7 GB — required by ReLiK cIE
#   ./scripts/download_data.sh --corpus    # +2.3 GB — required to reproduce paper §4.3 stats
# See src/benchmarks/README.md for each model's exact flag.

# 3. Run evaluation (reproduces paper results)
./scripts/run/evaluate.sh
# Without --kg downloaded, evaluate.sh emits a warning and runs anyway:
# 12 of 13 models score correctly; only relik-cie's Exists row is approximate.
# Download --kg + re-run (same command) for full relik-cie Exists scoring.

# 4. View results in notebooks (saved outputs included — no re-execution needed)
jupyter lab src/stats/
```

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
- Predictions from 13 benchmark models

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

## Dataset construction pipeline

The released `data/evaluation_set/` is the **frozen final artifact** of the pipeline
below. **Using or evaluating against the dataset does NOT require running any of these
stages** — `download_data.sh` + `evaluate.sh` is sufficient. The construction code is
shipped here for **transparency**, not as one-click reproduction; full from-scratch
reconstruction is a research-level effort.

```
Raw Wikidata dumps  ──→  [WD s01–s03]  ──→  KG snapshots + per-triple deltas
                                                      │
Raw Wikipedia dumps ──→  [WP s01–s03]  ──→  Entity descriptions + hyperlink history
                                                      │
                                                      ▼
                              [s03] Match textual passages ↔ KG-change deltas
                                                      │
                              [s04] Filter for KG-relevant snippets
                                                      │
                              [s05] LLM assessment (Llama 8B + Llama 405B w/ TGI)
                                                      │
                              [s06b → s07 → s07b] Reformat → dedup → 35K → 3.5K
                                                      │
                              [s09] Human annotation + agreement statistics
                                                      │
                              [Part 3] Quality control of LLM annotations  ◀── extension
                                                      │
                                                      ▼
                                          data/evaluation_set/   (released)
```

| Stage | Code in repo | Configs | Reviewer can run? | What's needed otherwise |
|---|:-:|:-:|:-:|---|
| Raw Wikipedia/Wikidata dump download | — | — | — | Fetch from [dumps.wikimedia.org](https://dumps.wikimedia.org/); ~hundreds of GB to TB |
| WD s01 (Java triple extraction) | ✅ `src/dataset/wikidata/` | placeholder | ❌ | Raw 7z dump + Java tooling |
| WD s02–03 / WP s01–03 (normalize, snapshots, hyperlinks) | ✅ `src/dataset/{wikidata,wikipedia}/` | placeholder | ❌ | s01 outputs; SLURM hardware |
| s03 textual delta snippets | ✅ `src/dataset/emerge/s03_*.py` | placeholder | ❌ | WD/WP outputs |
| s04 find interesting snippets | ✅ `src/dataset/emerge/s04_*.py` | placeholder | ❌ | s03 outputs |
| **s05 LLM annotation (Llama 8B + 405B, prompts v1)** | ✅ `src/dataset/emerge/s05_*.py` + `src/dataset/emerge/prompts/prompts_v1/` + `scripts/slurm/dataset/s05_*` | placeholder | ❌ | Llama 405B model + multi-H100 cluster + TGI inside Apptainer; **also note: the released `data/corpus/` is the post-s05 1.19M-instance output, so even with infra you'd need the s04-output candidate set (not redistributed) as input** |
| s06b/s07/s07b reformat/dedup/subsample | ✅ `src/dataset/emerge/{s06b,s07,s07b,s07c}_*.py` | placeholder | ❌ | s05 outputs |
| s09 human annotation + stats | ✅ `src/dataset/emerge/s09{a,b,c,d,e}_*.py` | partial (`s09e_*` is functional) | partial | `data/human_annotation/` (in `download_data.sh`); annotators for new annotation rounds |
| **Part 3 — Quality control of LLM annotations** | ✅ `scripts/dataset/` (4 CLI tools) | functional | **✅** | Runs on `data/corpus/` (`download_data.sh --corpus`); reproduces the **220-flagged-triples (0.0113%)** residual from the paper |
| **Paper §4.3 / Table 8 statistics** | ✅ `scripts/stats/` (3 CLI tools) | functional | **✅** | Runs on `data/corpus/`; reproduces 608K / 207K / 149K / 220K / 9.5K / 1.19M headline numbers |

**Why most stages aren't reviewer-runnable end-to-end:**

- **Storage scale.** The English Wikipedia history dump alone is hundreds of GB compressed; full Wikidata is similar. We don't redistribute these dumps via this repo or the HF dataset — they're public on [dumps.wikimedia.org](https://dumps.wikimedia.org/) for anyone who wants to attempt full reproduction.
- **LLM serving.** s05 is designed around Llama 405B served via TGI inside Apptainer on a multi-H100 SLURM cluster. The `scripts/slurm/dataset/s05_*` scripts document exactly how, but a single-machine reviewer setup cannot host the model.
- **Configs are placeholders.** All `config/dataset/...` JSONs use `/path/to/storage/...` strings as a structural template. Anyone running these stages must rewrite them for their own dump downloads + intermediate outputs.

**What's reproducible by reviewers right now**, on a regular workstation:
- The two ✅ rows above (Part 3 garbage QA + paper §4.3 stats) — both fully runnable, both reproduce documented paper numbers.
- All 12 of 13 baseline benchmarks (the 13th, `relik_cie`, additionally requires the entity index — see the heads-up under "Running benchmark models").
- The full `evaluate.sh` paper-results pipeline against the released `data/evaluation_set/`.

**Possible future additions** (not blocking; signaling direction): dump-download convenience script, uploads of more intermediate outputs (`data/corpus/` already covers part of this), and a single-node Llama-8B-only variant of s05 for low-resource users.

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

- **Python 3.10+** with conda
- **GPU**: 1x GPU with 16GB+ VRAM for evaluation (BERTScore, sentence-transformers)
- **RAM**: 32GB minimum. 180GB+ only if loading full KG snapshots for Exists operation evaluation of CIE models
- **API keys** (only for LLM-based benchmark models): Azure OpenAI and/or OpenAI

## Setup

### Step 1: Create conda environment

```bash
conda create -n emerge python=3.12
conda activate emerge

# PyTorch — default install (CPU on most systems). For GPU, pin the CUDA-matching
# build first per https://pytorch.org/ (e.g. `pip install torch torchvision torchaudio
# --index-url https://download.pytorch.org/whl/cu121` for CUDA 12.1) BEFORE the
# core.txt install below; pip will then see torch as already-satisfied.
pip install torch torchvision torchaudio

# Core dependencies (evaluation, statistics, merge)
pip install -r requirements/core.txt
```

After installing, download NLTK data:
```bash
python -c "import nltk; nltk.download('words'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

> **Benchmark models** require separate conda environments due to package conflicts.
> See `requirements/benchmarks/` for per-model setup instructions.
>
> **Dataset creation** has additional dependencies: `pip install -r requirements/dataset.txt`

### Step 2: Download the dataset

Download the EMERGE dataset from HuggingFace Hub:

```bash
./scripts/download_data.sh
```

This downloads the test set and annotation data into `data/`:

```
data/
├── evaluation_set/             Test set (3,500 instances + 13 model predictions)
│   ├── snapshot_2019-01-01/
│   │   ├── delta_2019-01-08.jsonl
│   │   └── ...
│   ├── snapshot_2020-01-01/ ... snapshot_2025-01-01/
│   └── (7 snapshots x 5 deltas x 100 instances = 3,500 total)
└── human_annotation/           Human annotation data for agreement statistics
    └── solved_disagreements.jsonl
```

To also download KG snapshots (~22GB, needed only for relik-cie Exists evaluation):

```bash
./scripts/download_data.sh --kg
```

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
