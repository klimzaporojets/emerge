# KG-aware Baseline

A targeted baseline that gives the **same GPT-5.1 used by EDC+** access to the reference
KG snapshot `G_t`, keeping the model fixed and varying only its access to KG structure.
It tests directly whether feeding KG context closes the gap that IE-only baselines leave, and
— because it canonicalizes its outputs against `G_t` — it is the only IE baseline able to
report **Executable-R** for EXISTS / ADD / DEPRECATE and to execute DEPRECATE at all.

Results (a Table-3-style ladder over three KG-context variants vs. EDC+ and ReLiK cIE) are in the
[main README → KG-aware baseline](../../README.md#kg-aware-baseline). This document is the runbook
to reproduce them.

## The three KG-context variants

| Variant | KG context supplied |
|---|---|
| `oracle` | gold entities + relation types (ADD, MINT+ADD, INFER) and gold existing triples (EXISTS, DEPRECATE) |
| `kg_rag@32` | top-`k=32` candidate entities, relation types and triples retrieved from `G_t` with ReLiK |
| `oracle+kg_rag` | `oracle` context plus the `kg_rag@32` retrieved distractors |

## Pipeline

Four stages. Each is config-driven (`--config`); run from the repo root with `PYTHONPATH=src`.

| # | Stage | Script | Conda env | GPU | Cost |
|---|---|---|---|---|---|
| 1 | **AUGMENT** — retrieve KG context per passage | `scripts/kg_context/build_augmented_evalset.py` | `relik` | optional | local, free |
| 2 | **COMPOSE** — render prompts from components | `scripts/kg_context/compose_kg_prompt.py` | `relik` | no | local, free |
| 3 | **PREDICT** — call GPT-5.1 on each prompt | `scripts/kg_context/predict_triples.py` | `edc` | no | **API $** |
| 4 | **SCORE** — Table-3 metrics (C / G-BERTScore-R / Executable-R) | `scripts/kg_context/score_kg_prompt.py` | `emerge` | yes | local, free |

**Three envs (they don't share, by design):**
- **AUGMENT / COMPOSE** need the **ReLiK retriever** → a `relik` env: `pip install -r requirements/benchmarks/relik.txt`. ReLiK's dependencies can conflict with the core env, so keep it in its own env (named `relik` here).
- **PREDICT** reuses EDC+'s `edc.utils.unified_llm_client.UnifiedLLMClient` → the **`edc`** env: `pip install -r requirements/benchmarks/edc-plus.txt` (has `openai`/`aiohttp`/`transformers`). PREDICT goes through `wrappers/kg_prompt/run.sh`, which does `conda activate "${KG_PROMPT_ENV:-edc}"` — so if your EDC+ env has a different name, just `export KG_PROMPT_ENV=<that-env>` instead of creating an `edc` env.
- **SCORE** uses the core **`emerge`** env (metrics stack: sentence-transformers, BERTScore).

See the [benchmarks README](../benchmarks/README.md) for per-env setup.

**AUGMENT resources & device.** AUGMENT holds three things in memory at once: the whole KG snapshot
(loaded as a dict, ~15–20 GB for a full year), and the ReLiK entity index for that year
(`documents.jsonl` ~7.7 GB + `embeddings.pt` ~10.4 GB). It processes **one snapshot year per
subprocess** (peak = one year), so peak memory is a **fixed cost independent of the number of
instances**. Set the retriever `device` in the AUGMENT config:
- `"device": "cpu"` — the embeddings sit in **CPU RAM**; budget **~64–96 GB RAM** (32 GB OOM-kills). No GPU needed.
- `"device": "cuda"` — the embeddings load onto the **GPU**, cutting CPU RAM to ~35–45 GB (so ~64 GB is plenty) and speeding up retrieval; needs 1 GPU.

Either works; pick per your hardware. SCORE needs 1 GPU (BERTScore / sentence-transformers);
COMPOSE and PREDICT are light.

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
# 1. AUGMENT  (relik env — the ReLiK retriever needs it)
conda activate relik
PYTHONPATH=src python scripts/kg_context/build_augmented_evalset.py \
    --config config/kg_context/20260725_relik_all_ops/config.json

# 2. COMPOSE  (relik env) — renders the prompt ladder; also carries the `predict` block PREDICT reads
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

### Tiny smoke test (verify the pipeline, ~cents)

A dedicated **`tiny`** config chain runs the full AUGMENT → COMPOSE → PREDICT → SCORE flow on just
**8 instances** — enough to confirm everything wires up and produces a ladder, without a real bill
(PREDICT is the only paid step: ~48 GPT-5.1 calls here).

```bash
conda activate relik
PYTHONPATH=src python scripts/kg_context/build_augmented_evalset.py --config config/kg_context/tiny_augment/config.json
PYTHONPATH=src python scripts/kg_context/compose_kg_prompt.py       --config config/kg_context/tiny_compose/config.json

conda activate edc
export OPENAI_API_KEY='<your-openai-key>'
PYTHONPATH=src python scripts/kg_context/predict_triples.py --config config/kg_context/tiny_compose/config.json --dry-run   # preview, free
PYTHONPATH=src python scripts/kg_context/predict_triples.py --config config/kg_context/tiny_compose/config.json            # ~cents

conda activate emerge
PYTHONPATH=src python scripts/kg_context/score_kg_prompt.py --config config/kg_context/tiny_score/config.json --phase all
cat output/kg_prompts_scored/tiny/ladder_table.csv
```

Expect a `ladder_table.csv` with sane (non-empty) numbers. On 8 instances these are **not** the
paper numbers — the point is only that the pipeline runs clean end-to-end. AUGMENT / COMPOSE / SCORE
can be re-run freely; only PREDICT hits the API.

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
| Tiny smoke test (8 instances) | `config/kg_context/tiny_{augment,compose,score}/config.json` |
| Retriever coverage (recall@k, optional) | `config/kg_context/20260726_coverage_{native,aida}/config.json` |

## Code

- `src/kg_context/augment.py` — gather KG context (oracle / kg_rag / pattern triples)
- `src/kg_context/retriever.py`, `kg_lookup.py`, `index_builder.py` — ReLiK retrieval + snapshot lookup
- `src/kg_context/prompt_templates_kg_aware/` — per-operation prompt templates + few-shot pools
- `src/benchmarks/wrappers/kg_prompt/` — the PREDICT runner (sends the composed prompt verbatim to GPT-5.1)
