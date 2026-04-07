# Evaluation Pipeline

This module scores benchmark model predictions against ground-truth EMERGE annotations.
It computes four families of metrics across all 5 TKGU operation types:

> **Note on internal naming:** The codebase uses different identifiers for TKGU operations:
> Exists = `x-triples`, Add = `e-triples`, Mint+Add = `ee-triples`,
> Infer = `ee-kg-triples`, Deprecate = `d-triples`.

| Metric | What it measures | GPU needed |
|--------|-----------------|------------|
| **Completeness** | Sentence-transformer cosine similarity (all-mpnet-base-v2) between predicted and GT triples | Yes |
| **Graph-Judge** | BERTScore + BLEU/ROUGE between predicted and GT triples, aggregated at graph level | Yes |
| **Entity Coverage** | How well predicted entities match GT entities (sentence-transformer + BERTScore) | Yes |
| **CIE Exact Match** | Set-based P/R/F1 on Wikidata QID triples (for models that produce QIDs) | No |

The pipeline reads a merged dataset (from `src/merge/`) containing both GT triples and model
predictions, and produces a `WikiEvalResult` pickle file that the statistics notebooks consume.

---

## Two scoring modes

The scoring behavior is controlled by `score_empty_predictions_as_zero` in the config:

- **`true` (fixed, correct):** Instances with no predictions are scored as P=0/R=0/F1=0 and
  included in the average. This is the correct behavior.
- **`false` (legacy):** Instances with no predictions are excluded from the average.
  Reproduces the original ICML submission numbers (scores are higher due to this exclusion).

---

## QID Exact-Match Metrics

The `cie_exact_match` metric computes **exact-match P/R/F1 on Wikidata QID triples**
(set comparison, no cosine similarity). Only applies to models that produce QIDs
(e.g. `relik-cie`).

Enable in config:
```json
"metrics_to_calculate": {
    "cie_exact_match": {
        "model_alias": "exact_match",
        "models_to_report": ["relik-cie"]
    }
}
```

Results are shown in a separate table after the main similarity-based table.

---

## How caching works

There is a **single cache file** (`.pkl`) controlled by the `cache_path` config parameter.
It stores the entire `WikiEvalResult` (preloaded data + computed metric scores).
This one file is used at two levels:

### Level 1 — `load_results_from_cache`

```json
"load_results_from_cache": true
```

When `true`, the pipeline **skips the preloader and all metric computation entirely** and
loads the cached `WikiEvalResult` from `cache_path` directly. Only the final table-building
code runs. Use this for fast re-runs after a full run has already completed.

### Level 2 — Incremental per-instance cache

Even with `load_results_from_cache: false`, if the `.pkl` file at `cache_path` already
exists, the pipeline loads it and:
- The **preloader** skips instances (hash_ids) already present in the cache.
- Each **metric** skips model/task/instance combinations already scored.
- After each metric finishes a model/task chunk, it **saves progress** back to the `.pkl`.

This is critical for expensive metrics (graph_judge, entity_coverage) — if the pipeline
crashes midway, restarting picks up where it left off instead of recomputing everything.

### What this means in practice

- **First run**: Set `load_results_from_cache: false`. Point `cache_path` to a **new file**
  (one that doesn't exist yet). The pipeline processes everything from scratch and saves
  the result to `cache_path`.
- **Re-runs (table tweaks, etc.)**: Set `load_results_from_cache: true`. The pipeline loads
  the cached result and just rebuilds the tables.
- **Adding a new model or new instances**: Set `load_results_from_cache: false` with the
  **same** `cache_path`. The incremental cache will skip already-done work and only compute
  the new model/instances.
- **After a code fix (e.g. `score_empty_predictions_as_zero`)**: Point `cache_path` to a
  **new file**. The old cache contains results computed with the old logic; the incremental
  cache would skip all those instances, so you must use a fresh path.

---

## How to run

From the repository root, with `conda activate emerge`:

```bash
export PYTHONPATH="$PWD/src"

python -u -m evaluation.s0x_evaluate_predictions \
    --config_file config/evaluation/s0x_evaluate_predictions/<config_name>/config.json \
    2>&1 | tee logs/evaluation.log
```

This requires a GPU for BERTScore and sentence-transformer metrics.

<details>
<summary>SLURM equivalent</summary>

```bash
sbatch scripts/slurm/evaluation/s0x_evaluate_predictions.sh \
    <config_file> <experiment_id>
```

The script requests 1 GPU, 180G memory, 18 CPUs, 4h time limit.
For cached re-runs (`load_results_from_cache: true`), no GPU is needed.
</details>

---

## Available configs

| Config | Scoring mode | KG snapshots | Description |
|--------|-------------|--------------|-------------|
| `20260324_all_models_with_zs_fixed_with_kg/` | Fixed (correct) | Yes | **Recommended.** All 13 models including zero-shot, correct scoring |
| `20260324_all_models_with_zs_legacy_with_kg/` | Legacy | Yes | All 13 models, reproduces original submission numbers |

**Without KG snapshots** (`snapshot_year_to_kg_file: null`): faster to run (~32GB RAM), but CIE models
like `relik-cie` cannot distinguish Exists from Add operations — all predictions go to
Add. Sufficient for all models except `relik-cie` Exists operations.

**With KG snapshots** (`20260324_*_with_kg`): loads KG snapshot TSVs into memory (~22GB
per snapshot, ~154GB total for 7 years). Required for correct `relik-cie` Exists
evaluation. Needs 180G+ RAM.

---

## Evaluation statistics (Jupyter notebooks)

Notebooks in `src/stats/evaluation/` generate paper tables and figures about **model
performance**. They load the `wiki_eval_result.pkl` files produced by the evaluation
pipeline above.

For the full list of notebooks, data dependencies, and configuration details, see
[`src/stats/README.md`](../stats/README.md).
