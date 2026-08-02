# KG-aware Baseline

A targeted baseline that gives the **same GPT-5.1 used by EDC+** access to the reference
KG snapshot \(G_t\), keeping the model fixed and varying only its access to KG structure.
It tests directly whether feeding KG context closes the gap that IE-only baselines leave, and
— because it canonicalizes its outputs against \(G_t\) — it is the only IE baseline able to
report **Executable-R** for EXISTS / ADD / DEPRECATE and to execute DEPRECATE at all.

Results (a Table-3-style ladder over three KG-context variants vs. EDC+ and ReLiK cIE) are in the
[main README → KG-aware baseline](../../README.md#kg-aware-baseline). This document is the runbook
to reproduce them.

## The three KG-context variants

| Variant | KG context supplied |
|---|---|
| `oracle` | gold entities + relation types (ADD, MINT+ADD, INFER) and gold existing triples (EXISTS, DEPRECATE) |
| `kg_rag@32` | top-\(k=32\) candidate entities, relation types and triples retrieved from \(G_t\) with ReLiK |
| `oracle+kg_rag` | `oracle` context plus the `kg_rag@32` retrieved distractors |

## Pipeline

Four stages. Each is config-driven (`--config`); run from the repo root with `PYTHONPATH=src`.

| # | Stage | Script | Conda env | GPU | Cost |
|---|---|---|---|---|---|
| 1 | **AUGMENT** — retrieve KG context per passage | `scripts/kg_context/build_augmented_evalset.py` | `emerge` | optional | local, free |
| 2 | **COMPOSE** — render prompts from components | `scripts/kg_context/compose_kg_prompt.py` | `emerge` | no | local, free |
| 3 | **PREDICT** — call GPT-5.1 on each prompt | `scripts/kg_context/predict_triples.py` | `edc` | no | **API $** |
| 4 | **SCORE** — Table-3 metrics (C / G-BERTScore-R / Executable-R) | `scripts/kg_context/score_kg_prompt.py` | `emerge` | yes | local, free |

**Why two envs?** AUGMENT/COMPOSE/SCORE run in the core `emerge` env (which has the ReLiK
retriever and the evaluation stack). PREDICT runs in the `edc` env because it reuses EDC+'s
`edc.utils.unified_llm_client.UnifiedLLMClient` — the `edc` env has `openai`/`aiohttp`/`transformers`.
The `edc` env cannot run ReLiK retrieval and the `emerge` env does not have the EDC+ client, so the
stages are deliberately split (see [benchmarks README](../benchmarks/README.md) for the env setup).

## Prerequisites (data)

All paths in the configs are **repo-relative** (`data/...`, `output/...`) — nothing points at an
absolute or cluster path. Fetch the data with the repo's downloader:

```bash
./scripts/download_data.sh --indices   # eval set + ReLiK indices
./scripts/download_data.sh --kg        # 7 yearly Wikidata KG snapshots (needed by AUGMENT + SCORE)
```

This provides:

- `data/evaluation_set/` — the 3,500-instance test set
- `data/kg_snapshots/kg_snapshot_{year}-01-01.tsv` — KG snapshots
- `data/indices/relik_entity_index/` (native), `data/indices/relik_edc_relation_indexes/` (relations),
  and `data/indices/relik_entity_index_aida/` (optional AIDA entity index)

**On Snellius**, don't copy the data — symlink each `data/*` entry to the cluster copy so the
repo-relative configs resolve unchanged, e.g.:

```bash
ln -sfn /path/to/cluster/kg_snapshots      data/kg_snapshots
ln -sfn /path/to/cluster/evaluation_set    data/evaluation_set
ln -sfn /path/to/cluster/relik_entity_index data/indices/relik_entity_index
# ...relation index, aida index as needed
```

## Ground truth = 405B-supported operations

A TKGU operation is ground truth only if Meta-Llama-3.1-405B assessed it as text-supported
(the `llm_assessment` flag) — this is exactly what Table 3 reports. AUGMENT/COMPOSE build oracle
context and few-shot demos only from 405B-supported operations, and SCORE evaluates only against
them, so the numbers are directly comparable to Table 3.

## Run

Representative chain (small, `n_instances=32` in the AUGMENT config — raise it, or set `hash_ids`,
to scale toward the full 3,500).

```bash
# 1. AUGMENT  (emerge env)
conda activate emerge
PYTHONPATH=src python scripts/kg_context/build_augmented_evalset.py \
    --config config/kg_context/20260725_relik_all_ops/config.json

# 2. COMPOSE  (emerge env) — renders the prompt ladder; also carries the `predict` block PREDICT reads
PYTHONPATH=src python scripts/kg_context/compose_kg_prompt.py \
    --config config/kg_context/20260725_compose_ladder/config.json

# 3. PREDICT  (edc env, costs API $) — GPT-5.1 via OpenAI
conda activate edc
export OPENAI_API_KEY='<your-openai-key>'
PYTHONPATH=src python scripts/kg_context/predict_triples.py \
    --config config/kg_context/20260725_compose_ladder/config.json
#   add --dry-run first to preview the commands without spending anything

# 4. SCORE  (emerge env; GPU + KG snapshots — a Snellius compute node, NOT the login node)
conda activate emerge
PYTHONPATH=src python scripts/kg_context/score_kg_prompt.py \
    --config config/kg_context/20260725_kg_prompt_ladder_score/config.json --phase all
```

SCORE writes the ladder to
`output/kg_prompts_scored/20260725_ladder_all_ops/ladder_table.csv`
(rows = op × condition; columns = C / G-BERTScore-R / Executable-R).

### Cheap smoke test (verify the pipeline, don't spend much)

PREDICT is the only paid step. To confirm the chain runs end-to-end without a large bill:

1. keep `n_instances` small (the representative AUGMENT config uses 32) — or set `sample.hash_ids`
   to a handful of ids;
2. run stages 1–2 (free), then stage 3 with `--dry-run` to inspect the commands, then a real
   stage-3 run on that small sample;
3. run stage 4 and confirm `ladder_table.csv` is produced with sensible numbers.

AUGMENT / COMPOSE / SCORE can be re-run freely; only re-run PREDICT when you actually need new
model outputs (it hits the API).

### Using Azure instead of OpenAI (optional)

The default is OpenAI-direct (`OPENAI_API_KEY`). The `UnifiedLLMClient` also supports Azure —
set the runner config's `backend` / `base_url` / `api_version` / `api_key_env` accordingly
(`config/benchmarks/s02_run_benchmarks/kg_prompt_gpt_5_1/config.json`), mirroring how the
Mistral / Azure-hosted baselines are configured.

## Configs

| Stage | Config |
|---|---|
| AUGMENT | `config/kg_context/20260725_relik_all_ops/config.json` |
| COMPOSE + PREDICT block | `config/kg_context/20260725_compose_ladder/config.json` |
| PREDICT runner (LLM/connection) | `config/benchmarks/s02_run_benchmarks/kg_prompt_gpt_5_1/config.json` |
| SCORE | `config/kg_context/20260725_kg_prompt_ladder_score/config.json` |
| Retriever coverage (recall@k, optional) | `config/kg_context/20260726_coverage_{native,aida}/config.json` |

## Code

- `src/kg_context/augment.py` — gather KG context (oracle / kg_rag / pattern triples)
- `src/kg_context/retriever.py`, `kg_lookup.py`, `index_builder.py` — ReLiK retrieval + snapshot lookup
- `src/kg_context/prompt_templates_kg_aware/` — per-operation prompt templates + few-shot pools
- `src/benchmarks/wrappers/kg_prompt/` — the PREDICT runner (sends the composed prompt verbatim to GPT-5.1)
