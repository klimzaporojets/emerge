# Paper Statistics (Jupyter Notebooks)

All paper tables and figures are generated from Jupyter notebooks, organized into
two subdirectories:

> **Note on internal naming:** The codebase uses different identifiers for TKGU operations:
> Exists = `x-triples`, Add = `e-triples`, Mint+Add = `ee-triples`,
> Infer = `ee-kg-triples`, Deprecate = `d-triples`.

```
src/stats/
├── evaluation/             Model performance tables and figures
│   ├── load_results.py     Shared module: pkl loading, aggregation, LaTeX generation
│   └── *.ipynb
└── dataset/                Dataset-level statistics and figures
    ├── other_datasets/     Scripts for computing stats of other IE datasets
    └── *.ipynb
```

For how to launch JupyterLab on HPC cluster, see the
[top-level README](../../README.md#how-to-run-the-notebooks-slurm).

---

## Evaluation statistics (`evaluation/`)

Notebooks that measure **model performance**. They load `wiki_eval_result.pkl` files
produced by the evaluation pipeline (see [`src/evaluation/README.md`](../evaluation/README.md)
for how to run evaluations and produce these pkls).

### Notebooks

| Notebook | What it produces | Paper location |
|----------|-----------------|----------------|
| `tables_results.ipynb` | **Table 1**: Completeness (C) + G-BERTScore-R (G-R) for all models. **Table 2**: zero-shot vs ICL LaTeX table. Also a full C/P/R/F1 variant. | Main paper |
| `table_cie_exact_match.ipynb` | **Table 3**: Executable-R (E-R) — the recall component of `cie_exact_match` (set-based QID exact match), reported for ReLiK cIE on EXISTS + ADD where canonical-ID match is feasible at inference time. The notebook also shows full P/R/F1. | Main paper |
| `figure_deltas_snapshots.ipynb` | **Main**: `plot_deltas_overlay_metrics_partial.pdf` + `plot_snapshots_overlay_metrics_partial.pdf` (G-R only, 3 TKGU ops). **Appendix**: full versions with C + G-R, all 5 TKGU ops | Main + Appendix |
| `figure_zs_vs_icl.ipynb` | `zs_vs_icl.pdf`: bar chart comparing EDC+ zero-shot vs in-context learning with delta annotations | Main paper |
| `appendix_table_qualitative_predictions.ipynb` | Qualitative analysis table: randomly selected passages with GT triples paired with best-matching predictions by BERTScore | Appendix |

### Shared module: `load_results.py`

All notebooks import from `src/stats/evaluation/load_results.py`, which provides:
- `load_results()` / `load_from_wiki_eval_result()` — load and filter `WikiEvalResult` pkl
- `make_agg_and_agg_open()` / `make_agg_all()` — aggregate metrics for table/plot generation
- `make_metrics_latex_table()` — generate publication-ready LaTeX with best/second-best highlighting
- Model name mappings, TKGU type to LaTeX macros (`\opexists`, `\opadd`, etc.)

### Data dependencies

Each notebook has a `PKL_PATH` variable in its config cell. To re-execute the notebooks,
you first need to run the evaluation pipeline (see [`src/evaluation/README.md`](../evaluation/README.md))
which produces a `wiki_eval_result.pkl` file. Point `PKL_PATH` to that file.

Available evaluation configs produce different pkl variants:

| Config | Description |
|--------|-------------|
| `20260324_all_models_with_zs_fixed_with_kg/` | All 13 models, fixed scoring, with KG snapshots (recommended) |
| `20260324_all_models_with_zs_legacy_with_kg/` | All 13 models, legacy scoring (reproduces paper tables) |

See [`src/evaluation/README.md`](../evaluation/README.md) for the full config list and
the difference between fixed vs legacy scoring.

**Note:** The notebooks include saved cell outputs, so you can view all results
without re-executing.

**KG snapshots and CIE Exists operations.**
The `20260217` configs have `snapshot_year_to_kg_file` set to `null` (light mode).
CIE models like `relik-cie` cannot distinguish Exists from Add operations without KG
snapshots. The `20260324_*_with_kg` configs fix this. See
[`src/evaluation/README.md`](../evaluation/README.md) for details.

---

## Dataset statistics (`dataset/`)

Notebooks that measure **dataset properties** (GT triples, annotation quality, KG
statistics). These do NOT depend on model predictions or the evaluation scoring fix.
For the dataset creation pipeline itself, see [`src/dataset/README.md`](../dataset/README.md).

### Notebooks

| Notebook | What it produces | Paper location |
|----------|-----------------|----------------|
| `table_dataset_comparison.ipynb` | LaTeX table comparing EMERGE with other IE datasets (WebNLG, T-REX, DocRED, etc.) | Main paper |
| `figure_tkgu_distribution.ipynb` | `pie_chart_nr_triples_both.pdf` — pie chart of TKGU operation distribution | Main paper |
| `table_dataset_statistics.ipynb` | Per-snapshot statistics table: instances, operations, KG entities/relation types/triples | Appendix |
| `appendix_figure_annotation_stats.ipynb` | `annotation_stats.pdf` — bar chart of LLM-supported vs all TKGU operations | Appendix |
| `appendix_table_annotation_agreement.ipynb` | Annotation agreement LaTeX table — Cohen's kappa, Fleiss' kappa, Krippendorff's alpha | Appendix |
| `appendix_figure_tkgu_distribution_deltas.ipynb` | `tkgu_distribution_deltas.pdf` — stacked bar chart of TKGU operations across delta weeks | Appendix |
| `appendix_table_top_triples.ipynb` | LaTeX tables of most frequent triples per TKGU operation with example passages | Appendix |

### Supporting files

| File | Purpose |
|------|---------|
| `other_datasets/*.py` | Scripts that download and compute stats for other IE datasets (WebNLG, T-REX, REBEL, DocRED, etc.). Run individually, e.g. `python other_datasets/webnlg_stats.py --lang en`. Results are hardcoded in `table_dataset_comparison.ipynb`. |

### Data dependencies

All paths below are on HPC cluster. Each notebook has configurable path variables at the top.

**Subsampled test set** (3,500 instances — used by most notebooks):
- Same `WikiEvalResult` pkl as evaluation stats (see [pkl table above](#data-dependencies))

**Complete dataset** (233K instances — used by pie chart, annotation stats, top triples, per-snapshot table):
- GT triples: `df_wiki_predictions_and_gt_cie.pkl`
- Instances: `df_instances_v13.pkl`
- These pkls contain the full 233K-instance dataset with GT triples and LLM assessments
  (no model predictions). They are too large to include in this repository and will be
  provided in the public release. The notebooks include saved cell outputs so results
  can be viewed without these files.

**Annotation data** (used by annotation agreement table):
- Included in this repository at `data/annotation/solved_disagreements.jsonl`
- Also used by `s09e_annotation_stats_for_paper.py`.

**KG snapshot TSVs** (used by per-snapshot table for triples + relation types):
- `/path/to/storage/wikidata-processing/output/experiments/s05_extract_wikidata_kg_snapshots/20250329/kg_snapshot_YYYY-01-01.tsv`
- Filtered to Wikipedia entities (`heads_in_wikipedia=True`)

**Wikipedia entity dictionaries** (used by per-snapshot table for entity counts):
- `/path/to/storage/wikipedia-processing/output/experiments/s08_extract_relik_index/20250910_slurm/dictionary_v3/YYYY-01-01-wiki_dictionary.jsonl`
- Line count = number of entities at that snapshot

### Notes

**Assessor naming differs between complete dataset and subsampled test set.**
The complete dataset stores assessor names WITH prompt_type suffix
(e.g. `Meta-Llama-3.1-8B_triple_assertion`). The evaluation test set stores them
WITHOUT suffix (e.g. `Meta-Llama-3.1-405B_prompt_v1`). The notebooks handle this
difference automatically.

**KG relation types: filtered vs unfiltered.**
The KG snapshot TSVs are filtered to Wikipedia entities, so they contain fewer
relation types (962-1,351) than the full Wikidata KG. The `table_dataset_statistics.ipynb`
notebook has an optional cell for computing unfiltered relation types (requires 128GB RAM).
