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
conda create -n emerge python=3.10 -y && conda activate emerge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements/core.txt

# 2. Download the dataset
./scripts/download_data.sh

# 3. Run evaluation (reproduces paper results)
./scripts/run/evaluate.sh

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
├── download_data.sh            Download dataset from HuggingFace Hub
├── run/                        Non-SLURM entry points (evaluate, benchmarks, etc.)
└── slurm/                      Optional SLURM sbatch scripts (for HPC clusters)
```

## Requirements

- **Python 3.10+** with conda
- **GPU**: 1x GPU with 16GB+ VRAM for evaluation (BERTScore, sentence-transformers)
- **RAM**: 32GB minimum. 180GB+ only if loading full KG snapshots for Exists operation evaluation of CIE models
- **API keys** (only for LLM-based benchmark models): Azure OpenAI and/or OpenAI

## Setup

### Step 1: Create conda environment

```bash
conda create -n emerge python=3.10
conda activate emerge

# PyTorch (adjust CUDA version to match your system; see https://pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

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

## Two scoring modes

| Mode | `score_empty_predictions_as_zero` | Description |
|------|----------------------------------|-------------|
| **Fixed** (correct) | `true` | Instances with no predictions scored as P=0/R=0/F1=0 |
| **Legacy** | `false` | Instances with no predictions excluded from averages (reproduces original submission) |

## Key documentation

| File | What it covers |
|------|---------------|
| [`src/evaluation/README.md`](src/evaluation/README.md) | Evaluation pipeline: metrics, caching, scoring modes |
| [`src/benchmarks/README.md`](src/benchmarks/README.md) | All 13 benchmark models: architectures, APIs, configs |
| [`src/dataset/README.md`](src/dataset/README.md) | Full dataset creation pipeline (Wikidata + Wikipedia + EMERGE) |
| [`src/stats/README.md`](src/stats/README.md) | Paper statistics: notebook inventory, data dependencies |

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
