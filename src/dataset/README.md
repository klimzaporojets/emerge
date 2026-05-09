# Dataset Creation Pipeline

This module contains the complete pipeline for creating the EMERGE dataset, from raw
Wikidata/Wikipedia dumps to the final 3,500-instance evaluation set used in the paper.

> **Status of this pipeline:** The complete EMERGE construction pipeline — *every*
> stage from raw Wikidata/Wikipedia dump processing through LLM assessment, deduplication,
> human annotation, and quality control — is shipped here for **transparency**. The whole
> chain is the EMERGE contribution. What's *not* in the repo are the **inputs** (raw
> history dumps ~TB; Llama 405B model weights) and the **HPC infrastructure** the heavy
> stages assume (multi-H100 SLURM cluster running TGI inside Apptainer). Reviewers using
> or evaluating against the released `data/evaluation_set/` do NOT need to run any of
> these stages.
>
> ### Pipeline stages (code in repo, every stage)
>
> All stages below have full Python/Java code in `src/dataset/`, SLURM submitters in
> `scripts/slurm/`, and configs (placeholder paths) in `config/dataset/`. Listed in
> execution order:
>
> - **WD s01** — Wikidata triple extraction from raw 7z dump:
>   `src/dataset/wikidata/java/wikidata/MainConcurrentWikidata.java` (Java);
>   SLURM `scripts/slurm/wikidata/s01_*.sh`.
> - **WD s02–s03** — Normalize entities/redirects, build per-triple deltas, extract
>   KG snapshots: `src/dataset/wikidata/python/{s02_normalize_history_graph,s02_normalize_entity_creation_date,s03_get_kg_snapshot,s03_get_deltas}.py`;
>   SLURM `scripts/slurm/wikidata/s02_*.sh`, `s03_*.sh`.
> - **WP s01** — Wikipedia title-change + hyperlink history extraction from raw XML:
>   `src/dataset/wikipedia/{s01_history_links_extraction,s01_extract_title_changes}.py`;
>   SLURM `scripts/slurm/wikipedia/s01_*.sh`.
> - **WP s02–s03** — Wikipedia normalize + entity-description extraction at snapshot
>   timestamps: `src/dataset/wikipedia/{s02_normalize_entity_creation_date,s02b_normalize_history_graph,s03_extract_entity_descriptions}.py`;
>   SLURM `scripts/slurm/wikipedia/s02_*.sh`.
> - **emerge s03–s04** — Match WP passages with WD deltas → filter for KG-relevant
>   snippets: `src/dataset/emerge/{s03_obtain_textual_delta_snippets_v8,s04_find_interesting_snippets_v3}.py`.
> - **s05 LLM assessment (Llama 8B + Llama 405B, prompts v1):**
>   - Python runner: `src/dataset/emerge/s05_generate_dataset_with_llm_v8.py`
>   - SLURM 405B submitter: `scripts/slurm/dataset/s05_generate_dataset_with_llm_v9_llama405b.sh`
>   - SLURM 8B variant: `scripts/slurm/dataset/s05_generate_dataset_with_llm_v9_llama8b.sh`
>   - Fault-tolerant TGI/Apptainer restart loop: `scripts/slurm/dataset/s05_restart_apptainer_llama400b_v5.sh`
>   - Prompt templates: `src/dataset/emerge/prompts/prompts_v1/` (single/multi assert + deprecate)
>   - Configs: `config/dataset/emerge/s05_generate_dataset_with_llm/{20251216_v8_llama_405b_v1, 20260502_v8_llama_405b_v1_complete_dataset}/`
> - **s06b/s07/s07b/s07c** — Reformat → dedup → 35K subsample → 3.5K subsample:
>   `src/dataset/emerge/{s06b_refactor_final_format,s07_reduce_duplicates_v6,s07b_subsample_dataset,s07c_verify_duplicates}.py`.
> - **s08** — Extract ReLiK entity index from Wikipedia:
>   `src/dataset/emerge/s08_*.py`; SLURM `scripts/slurm/dataset/s08_extract_relik_dictionary.sh`.
> - **s09** — Human annotation + agreement statistics:
>   `src/dataset/emerge/s09{a,b,c,d,e}_*.py`.
> - **Part 3 — Garbage QA / quality control of LLM annotations:**
>   `scripts/dataset/{find_garbage_clusters,build_reinput_for_garbage,merge_reinput_into_dataset,inspect_cluster_neighbors}.py`.
>   See the Part 3 section below.
>
> ### What you can practically execute (no HPC, no Llama serving)
>
> The released artifacts only include certain checkpoints in the chain — not the raw
> dumps and not s04 output. So **practically**, on a regular workstation, the
> reviewer-runnable subset is:
>
> - **Part 3 garbage QA** on `data/corpus/` (downloaded via
>   `./scripts/download_data.sh --corpus`). Validated reproduces the paper's
>   220-flagged-triples residual (0.0113%).
> - **Paper §4.3 / Table 8 statistics** via `scripts/stats/*.py` on `data/corpus/`.
>   Reproduces the 608K / 207K / 149K / 220K / 9.5K / 1.19M headline numbers.
> - **Annotation agreement** on `data/human_annotation/` via
>   `./scripts/run/annotation_agreement.sh`.
> - **Benchmark re-run** of any of the 13 baselines on `data/evaluation_set/` (most
>   need API keys or GPU; `relik_cie` additionally needs the entity index — see top
>   README and `src/benchmarks/README.md`).
> - **Full evaluation** (`./scripts/run/evaluate.sh`) against `data/evaluation_set/`
>   to reproduce paper numbers.
>
> ### What you cannot realistically run as a reviewer
>
> - **WD s01–s03 / WP s01–s03**: needs raw history dumps from
>   [dumps.wikimedia.org](https://dumps.wikimedia.org/) — specifically the
>   `wikidatawiki-YYYYMMDD-pages-meta-history*.7z` and
>   `enwiki-YYYYMMDD-pages-meta-history*.7z` series. Hundreds of GB to TB; days of
>   processing on an HPC node.
> - **emerge s03–s04**: depends on WD/WP s01–s03 outputs, which we don't
>   redistribute.
> - **s05 LLM assessment**: requires Llama 405B served via TGI inside Apptainer on a
>   multi-H100 SLURM cluster. **Note:** `data/corpus/` is the **post-s05** 1.19M
>   LLM-labelled corpus — already past this stage, so it can't be used as input to
>   re-run s05. To run s05 you'd need the s04-output candidate set (~1.19M unlabelled
>   snippets), which we don't redistribute.
> - **s06b–s09**: depends on s05 output; while `data/corpus/` is post-s05, none of
>   s06b–s09 has been smoke-tested for public use against it.
>
> **Configs are placeholders:** every JSON under `config/dataset/` uses
> `/path/to/storage/...` strings as a structural template. Rewriting them for your
> environment is the gating step before running anything beyond the reviewer-runnable
> subset above.
>
> See the top-level README's "Dataset construction pipeline" section for the per-stage
> status matrix at-a-glance.

> **Note on internal naming:** The codebase uses different identifiers for TKGU operations:
> Exists = `x-triples`, Add = `e-triples`, Mint+Add = `ee-triples`,
> Infer = `ee-kg-triples`, Deprecate = `d-triples`.

The pipeline has three major components:

1. **Wikidata pipeline** (`src/dataset/wikidata/`) — Extracts the full triple edit history from
   Wikidata dumps, normalizes entities and redirects, and produces KG snapshots at specific timestamps.

2. **Wikipedia pipeline** (`src/dataset/wikipedia/`) — Extracts hyperlink history, entity creation
   dates, title changes, and entity descriptions from Wikipedia dumps.

3. **EMERGE dataset construction** (`src/dataset/emerge/`) — Combines Wikidata deltas with Wikipedia
   text passages to identify textual evidence for knowledge graph changes. Includes LLM-based
   assessment, deduplication, subsampling, and human annotation.

## End-to-end data flow

```
Raw Wikidata dumps ──→ [WD s01-s03] ──→ KG snapshots + deltas
                                              │
Raw Wikipedia dumps ─→ [WP s01-s03] ──→ Entity descriptions + hyperlink history
                                              │
                                              ▼
                              [s03] Textual delta snippets (match text ↔ KG changes)
                                              │
                              [s04] Find interesting snippets (filter by KG)
                                              │
                              [s05] LLM assessment (Llama 8B + 405B)
                                              │
                              [s06b] Reformat into TKGU format
                                              │
                              [s07] Deduplicate passages
                                              │
                              [s07b] Subsample (35K → 3.5K)
                                              │
                              [s09] Human annotation + agreement stats
                                              │
                              [s05 v9] Final LLM re-assessment (405B prompt_v1)
                                              │
                              [merge] Combine dataset + assessments + predictions
                                              │
                              [evaluation] Score all models
```

---

## Part 1: Temporal knowledge graph snapshots

**Goal**: given a timestamp T, reconstruct the Wikidata knowledge graph as it existed at time T,
optionally filtered to entities that have a Wikipedia page.

The pipeline has two parallel tracks — **Wikidata** (KG triples) and **Wikipedia** (hyperlinks, entity
creation dates) — that merge at step 03 to produce the final snapshot.

## Wikidata pipeline

| Step | What it does | Input | Output |
|------|-------------|-------|--------|
| 01   | Extract full edit history of every triple | Wikidata meta-history 7z dump | Per-triple add/delete timeline + temporal qualifiers |
| 02a  | Normalize entity creation dates (resolve redirects) | Step 01 entities + Wikidata SQL dumps (page, redirect) | Single file: entity → earliest creation timestamp |
| 02b  | Normalize history graph (resolve redirects, filter unstable edits, merge duplicates) | Step 01 triples + Wikidata SQL dumps | Cleaned per-triple history with stable add/delete timeline |
| 03   | Extract KG snapshot at timestamp T | Step 02b output + timestamp(s) | TSV of all triples active at time T |

## Wikipedia pipeline (English)

These steps run in parallel with Wikidata and feed into step 03 via the `heads_in_wikipedia` /
`tails_in_wikipedia` filtering and `wpedia_entity_creation_date_path` config fields.

| Step | What it does | Input | Output |
|------|-------------|-------|--------|
| W01a | Extract page title changes from Wikipedia logging dump | Wikipedia XML logging dump + page SQL table | TSV of page moves (page_id, old_title, new_title, timestamp) |
| W01b | Extract historical hyperlinks from Wikipedia meta-history dump | Wikipedia 7z meta-history dump + SQL tables | Per-page link history + entity stats + title changes |
| W02a | Normalize entity creation dates (resolve redirects) | W01b entity stats + Wikipedia SQL dumps | Single file: entity QID → earliest creation timestamp |
| W02b | Normalize history graph (resolve redirects, filter unstable edits) | W01b link history + W01a/W01b title changes + Wikipedia SQL dumps | Cleaned per-link history |
| W03  | Extract entity descriptions at snapshot timestamps | Wikipedia meta-history 7z dump + W01a/W01b title changes + W02a entity dates + Wikidata s02a creation dates | Per-snapshot JSONL with entity title + description |


## step 01 - wikidata triple extraction (Java)

Reads Wikidata full meta-history XML dump (7z-compressed files) and, for every triple, records
the complete sequence of when it was added (A) or deleted (D) across all revisions.
Also extracts temporal qualifiers (e.g. P580=start time, P582=end time) attached to each triple.

**Output format** (tab-separated, one row per triple):

Triples file (`generated_history_triples/*.csv`):
```
subject  relation  object  type    timestamps
Q42      P569      +1879-03-14T00:00:00Z   time    1224592274000:A,
Q42      P27       Q183    entity  1224592274000:A,1350012345000:D,1350112345000:A,P580:Y2005MM1D1:9,
```
- `timestamps` column: comma-separated entries of two kinds:
  - `<epoch_ms>:A` or `<epoch_ms>:D` — the triple was Added or Deleted at that revision timestamp
  - `<qualifier_property>:<parsed_time>:<precision>` — a temporal qualifier attached to the triple (e.g. `P580:Y2005MM1D1:9`)

Entities file (`generated_history_entities/*.csv`):
```
entity   first_seen_timestamp
Q42      1224592274000
```

### How to run

Compile:
```
cd src/dataset/wikidata
mvn clean compile assembly:single
```

Run locally (salloc on slurm):
```
java -cp src/dataset/wikidata/target/wikidata-1.0-SNAPSHOT-jar-with-dependencies.jar \
    wikidata.MainConcurrentWikidata \
    /path/to/storage/wiki-dump-downloader/experiments/s01_download_wikidata_dump/20250201/metahistory7zdump \
    /path/to/storage/wikidata-processing/output/experiments/s01_history_links_extraction/20260311/generated_history_triples/ \
    /path/to/storage/wikidata-processing/output/experiments/s01_history_links_extraction/20260311/generated_history_entities/ \
    32
```

sbatch (slurm, 20260311 run):
```
sbatch scripts/slurm/wikidata/s01_wikidata_triple_extraction.sh \
  /path/to/storage/wiki-dump-downloader/experiments/s01_download_wikidata_dump/20250201/metahistory7zdump \
  /path/to/storage/wikidata-processing/output/experiments/s01_history_links_extraction/20260311/generated_history_triples/ \
  /path/to/storage/wikidata-processing/output/experiments/s01_history_links_extraction/20260311/generated_history_entities/ \
  32 \
  logs/s01_wdata_triple_extraction_20260311.log
```


## step 02a - normalize entity creation dates (Python)

Step 01 produces per-file entity creation timestamps, but an entity can appear in multiple files
(getting the minimum across them), and Wikidata redirects mean two different QIDs can refer to the
same entity. This step consolidates all entity files into a single file where each entity gets the
earliest creation timestamp, accounting for redirects.

Requires two Wikidata SQL dump files (gzipped):
- `page.sql.gz` — maps page IDs to QIDs
- `redirect.sql.gz` — maps redirected page IDs to target QIDs

**Config file** (JSON) keys:
- `path_extracted_entities`: directory with step 01 entity CSVs
- `path_wikidata_page`: path to `page.sql.gz`
- `path_redirects`: path to `redirect.sql.gz`
- `caches_dir`: directory for pickle caches (speeds up reruns)
- `output_dir_data`: output directory

**Output format** (`generated_entities/output_all_entities.csv`, tab-separated):
```
Q42     1224592274000
Q183    1227450000000
```
Each row: entity QID and its earliest creation timestamp (epoch ms), with redirects resolved.

### How to run

Run locally:
```
export PYTHONPATH="$PWD/src"
python -u -m dataset.wikidata.python.s02_normalize_entity_creation_date \
    --config_file config/dataset/wikidata/s02_normalize_entity_creation_date/20250320/config.json \
    --debug_max_size_tables -1
```

sbatch:
```
sbatch \
  --cpus-per-task=1 \
  --mem=32G \
  scripts/slurm/wikidata/s02_normalize_entity_creation_date.sh \
  config/dataset/wikidata/s02_normalize_entity_creation_date/20250320/config.json \
  logs/s02_normalize_entity_creation_date.log
```


## step 02b - normalize history graph (Python)

Step 01 produces per-file triple histories, but the same (subject, property, object) triple can appear
in multiple files (because the subject entity was split across dump files), and Wikidata redirects mean
two QIDs can refer to the same entity. This step:

1. **Resolves redirects** on both subject and object QIDs
2. **Filters unstable edits** — changes that are reverted within `stability_span` hours (default: 168h = 7 days) are removed
3. **Merges duplicate triples** — when the same triple appears from different source QIDs (due to redirects), their histories are merged chronologically

This step must be run **3 times** in sequence:

### Sub-step 1: First pass — normalize each file independently

Processes each step 01 triples file, resolves redirects, and filters unstable edits.
Output goes to `s02a_generated_history_triples_filtered/` (one file per input file).

```
sbatch scripts/slurm/wikidata/s02_normalize_history_graph.sh \
  config/dataset/wikidata/s02_normalize_history_graph/20250320/config.json \
  logs/s02_normalize_history_graph_20250320.log
```

### Sub-step 2: Concatenate and sort

After redirect resolution in sub-step 1, the same subject QID can appear in different output files.
To merge them, all files are concatenated and sorted so rows with the same subject are contiguous.
This requires a machine with enough RAM (~128G).

sbatch:
```
sbatch scripts/slurm/wikidata/s02_cat_sort.sh \
  /path/to/storage/wikidata-processing/output/experiments/s02_normalize_history_graph/20250320
```

Or interactively via salloc:
```
salloc -p rome -n 1 --ntasks-per-node 1 --cpus-per-task 1 -t 5:00:00 --mem=128G
```

Then run:
```
OUTPUT_BASE=/path/to/storage/wikidata-processing/output/experiments/s02_normalize_history_graph/20250320

mkdir -p $OUTPUT_BASE/s02b_cat/

cat $OUTPUT_BASE/s02a_generated_history_triples_filtered/* > $OUTPUT_BASE/s02b_cat/merged_file.csv

sort -S 120G --parallel=5 $OUTPUT_BASE/s02b_cat/merged_file.csv \
  -o $OUTPUT_BASE/s02b_cat/merged_file_sorted.csv

rm $OUTPUT_BASE/s02b_cat/merged_file.csv
```

### Sub-step 3: Second pass — merge duplicates from sorted file

Runs the same script again, but on the sorted concatenated file. Since rows are now sorted by subject,
the script can detect and merge triples that were split across files due to redirects.

```
sbatch scripts/slurm/wikidata/s02_normalize_history_graph.sh \
  config/dataset/wikidata/s02_normalize_history_graph/20250320/config_sorted.json \
  logs/s02_normalize_history_graph_20250320_sorted.log
```

**Config file** keys:
- `path_extracted_history_triples`: input triples directory (step 01 output for sub-step 1, `s02b_cat/` for sub-step 3)
- `path_wikidata_page`, `path_redirects`: Wikidata SQL dumps for redirect resolution
- `caches_dir`: pickle cache directory
- `output_dir_data`: output directory
- `stability_span`: hours before an edit is considered stable (default: 168)
- `timestamp_precision`: `"milliseconds"` or `"seconds"`

**Final output** (`s02c_.../s02a_generated_history_triples_filtered/merged_file_sorted.csv`, tab-separated):
```
subject  relation  object  type  timestamps  anchor_qids
42       P27       183     wikibase-entityid  1224592274000:A,1350012345000:D,P580:Y2005MM1D1:9  {42}
```
Same format as step 01 but with redirects resolved (QIDs are integers without `Q` prefix),
unstable edits filtered, and an extra column listing the original QIDs that mapped to this subject.


## step 03 - extract KG snapshot at timestamp (Python)

This is the final step that achieves the pipeline's goal: given a timestamp T, extract all Wikidata
triples that were active at that time.

It loads the normalized triple histories from step 02b into PyG tensors (with caching), then for each
requested timestamp applies a simple mask: a triple is active at time T if it was added before T and
either deleted after T or never deleted.

Supports multiple timestamps per run — the graph is loaded once (expensive, ~128GB RAM) and each
snapshot extraction is fast.

**Config file** keys:
- `path_extracted_history_triples_wdata`: step 02b output directory
- `wdata_entity_creation_date_path`: step 02a output (entity creation dates)
- `path_property_labels`: property label TSV (for relation IDs)
- `caches_dir`: directory for PyG tensor cache (speeds up reruns)
- `output_dir_data`: output directory
- `precision_wdata`: `"milliseconds"` or `"seconds"`
- `snapshot_timestamps`: list of dates, e.g. `["2023-01-01", "2024-01-01"]`
- `heads_in_wikipedia` / `tails_in_wikipedia`: if true, only include triples where head/tail has a Wikipedia page (default: false = include all)

**Output** (one directory per timestamp):
```
snapshot_2024-01-01/
  triples.tsv       # tab-separated: subject_qid  relation_id  object_qid
  metadata.json     # timestamp, nr_triples, commit_hash, config
```

`triples.tsv` example:
```
Q42     P27     Q183
Q42     P569    Q515
```

### How to run

Run locally:
```
export PYTHONPATH="$PWD/src"
python -u -m dataset.wikidata.python.s03_get_kg_snapshot \
    --config_file config/dataset/wikidata/s03_get_kg_snapshot/20250320/config.json \
    --debug_nr_triples -1
```

sbatch (slurm):
```
sbatch scripts/slurm/wikidata/s03_get_kg_snapshot.sh \
  config/dataset/wikidata/s03_get_kg_snapshot/20250320/config.json \
  logs/s03_get_kg_snapshot_20250320.log
```


---

## Wikipedia step W01a - extract title changes (Python)

Extracts page move (rename) events from Wikipedia's official XML logging dump. This is the new
approach — previously title changes were parsed from article revision history (now commented out
in `s01_history_links_extraction.py`).

**Output**: `page_title_changes.tsv` with columns: `page_id  old_title  new_title  unix_timestamp  date`

### How to run

```
export PYTHONPATH="$PWD/src"
python -u -m dataset.wikipedia.s01_extract_title_changes \
    --config_file config/dataset/wikipedia/s01_extract_title_changes/20251101_slurm_english/config.json
```

sbatch (English):
```
sbatch \
  --cpus-per-task=1 \
  --mem=8G \
  scripts/slurm/wikipedia/s01_extract_title_changes.sh \
  config/dataset/wikipedia/s01_extract_title_changes/20251101_slurm_english/config.json \
  logs/s01_extract_title_changes_20251101_english.log
```

sbatch (example for Danish):
```
sbatch scripts/slurm/wikipedia/s01_extract_title_changes.sh \
  config/dataset/wikipedia/s01_extract_title_changes/20260207_slurm_multilingual/s01_config_20251101_danish.json \
  logs/s01_extract_title_changes_20251101_danish.log
```


## Wikipedia step W01b - extract historical hyperlinks (Python)

Reads Wikipedia full meta-history XML dump (7z-compressed) and extracts the complete hyperlink
history for every article: which pages each article linked to, and when those links were added/removed.
Also extracts entity creation timestamps and title changes (legacy, now superseded by W01a).

Uses multiprocessing — the number of threads should match the number of 7z files in the dump.

**Output** (subdirectories of `output_dir_data`):
- `wikipedia_history/` — per-file link history CSVs
- `entity_stats/` — per-file entity creation timestamps
- `entity_title_changes/` — per-file title changes (legacy)

### How to run (English Wikipedia, slurm)

```
sbatch \
  --cpus-per-task=64 \
  --mem=196G \
  scripts/slurm/wikipedia/s01_history_links_extraction.sh \
  64 \
  config/dataset/wikipedia/s01_history_links_extraction/20251101_slurm_english/config.json \
  logs/s01_history_links_extraction_20251101_english.log
```


## Wikipedia step W02a - normalize entity creation dates (Python)

Same logic as Wikidata step 02a but for Wikipedia entities. Consolidates entity creation timestamps
from step W01b, resolving redirects so each entity gets its earliest creation timestamp.

### How to run

```
sbatch scripts/slurm/wikipedia/s02_normalize_entity_creation_date.sh \
  config/dataset/wikipedia/s02_normalize_entity_creation_date/20250320/config.json \
  logs/s02_wpedia_normalize_entity_creation_date_20250320.log
```


## Wikipedia step W02b - normalize history graph (Python)

Same 3-sub-step pattern as Wikidata step 02b: normalize each file, cat+sort, re-normalize.
Resolves redirects, filters unstable edits (reverted within `stability_span` hours), and uses
title change data from step W01a/W01b to correctly track pages across renames.

### Sub-step 1: First pass

```
sbatch scripts/slurm/wikipedia/s02_normalize_history_graph.sh \
  config/dataset/wikipedia/s02_normalize_history_graph/20250320/config.json \
  logs/s02_wpedia_normalize_history_graph_20250320.log
```

### Sub-step 2: Concatenate and sort

sbatch:
```
sbatch scripts/slurm/wikipedia/s02_cat_sort.sh \
  /path/to/storage/wikipedia-processing/output/experiments/s02_history_links_normalization/20250320
```

Or interactively via salloc:
```
salloc -p rome -n 1 --ntasks-per-node 1 --cpus-per-task 1 -t 5:00:00 --mem=128G
```

```
OUTPUT_BASE=/path/to/storage/wikipedia-processing/output/experiments/s02_history_links_normalization/20250320

mkdir -p $OUTPUT_BASE/wikipedia_history_filtered_cat/

cat $OUTPUT_BASE/wikipedia_history_filtered/* > $OUTPUT_BASE/wikipedia_history_filtered_cat/merged_file.csv

sort -S 120G --parallel=5 $OUTPUT_BASE/wikipedia_history_filtered_cat/merged_file.csv \
  -o $OUTPUT_BASE/wikipedia_history_filtered_cat/merged_file_sorted.csv

rm $OUTPUT_BASE/wikipedia_history_filtered_cat/merged_file.csv
```

### Sub-step 3: Second pass on sorted file

```
sbatch scripts/slurm/wikipedia/s02_normalize_history_graph.sh \
  config/dataset/wikipedia/s02_normalize_history_graph/20250320/config_sorted.json \
  logs/s02_wpedia_normalize_history_graph_20250320_sorted.log
```

**Final output**: `.../normalized_filtered_sorted_cat/wikipedia_history_filtered/merged_file_sorted.csv`

This is referenced by Wikidata step 03's config as `path_extracted_history_triples_wpedia`.


## Wikipedia step W03 - extract entity descriptions at snapshot timestamps (Python)

Extracts Wikipedia entity descriptions (first ~255 tokens of cleaned article text) at specific
snapshot timestamps. For each Wikipedia page, finds the revision that was active at each configured
snapshot date, cleans the wikitext, and writes the result as JSONL.

This step uses a **two-component architecture**:

1. **API server** (`s03_API_wiki_mapping.py`) — FastAPI app that loads Wikipedia SQL tables
   (page titles, redirects, QID mappings, title changes) and Wikidata entity creation dates into
   memory. Provides endpoints for the extractor to resolve page IDs to QIDs and check creation
   timestamps.

2. **Main extractor** (`s03_extract_entity_descriptions.py`) — Multiprocessing SAX parser that
   reads Wikipedia 7z meta-history dumps in parallel, calls the API to get each page's QID, then
   for each revision determines which snapshots fall within its time range.

**Dependencies**:
- W01a/W01b output: `path_extracted_title_changes` (title change data)
- W02a output: `qids_to_page_ids` (Wikipedia entity creation dates with page IDs)
- Wikidata s02a output: `qids_to_creation_date` (Wikidata entity creation dates)
- Raw Wikipedia dump: meta-history 7z files, page/redirect/page_props SQL tables

**Output** (per snapshot, inside `output_dir/dictionary/<snapshot_date>/`):
```
dictionary_<7z_filename>.jsonl
```

Each line is a JSON object:
```json
{"text": "Article Title", "qid": "Q42", "page_id": "123", "revision_id": 456,
 "revision_timestamp": 1234567890, "revision_date": "2023-01-15T10:30:00Z",
 "snapshot_date": "2023-01-01", "snapshot_timestamp": 1672531200.0,
 "metadata": {"definition": "First ~255 tokens of cleaned article text..."}}
```

After extraction, concatenate per-snapshot files:
```
cat output_dir/dictionary/2024-01-01/* > output_dir/dictionary_v2/2024-01-01-wiki_dictionary.jsonl
```

### How to run (English Wikipedia, slurm)

The sbatch script starts the API server(s), waits for them to load, then runs the extractor:

```
sbatch scripts/slurm/wikipedia/s03_extract_entity_descriptions.sh \
  config/dataset/wikipedia/s03_extract_entity_descriptions/20251101_slurm_english/config.json \
  20251101 8000 1
```

Arguments: `<config_file> <experiment_id> <start_port> <num_api_instances>`

To run locally (after starting the API manually):
```
# Terminal 1: start the API server
export PYTHONPATH="$PWD/src"
python -u -m dataset.wikipedia.s03_API_wiki_mapping \
    --config_file config/dataset/wikipedia/s03_extract_entity_descriptions/20251101_slurm_english/config.json \
    --debug_size -1 \
    --api_port 8000

# Terminal 2: run the extractor
export PYTHONPATH="$PWD/src"
python -u -m dataset.wikipedia.s03_extract_entity_descriptions \
    --config_file config/dataset/wikipedia/s03_extract_entity_descriptions/20251101_slurm_english/config.json \
    --nr_threads_processor 4 \
    --api_ports_wiki_mapping 8000
```


---

## Part 2: EMERGE dataset construction pipeline

These scripts transform raw LLM-assessed triples into the final evaluation dataset.
Located in `src/dataset/emerge/`.

### Data flow

```
s05_v4 (early LLM assessment, llama400b)
  → s06b (reformat into TKGU format, generate hash IDs)
  → s05_v8 (8B assertion) → s05_v8 (405B deprecation)
  → s07 (deduplicate by textual similarity + triple-level dedup)
  → s07b (stratified subsample: full 35K, then tiny 3.5K from full)
  → s07c (verify no remaining duplicates — read-only check)

After s07b (full 35K):
  → s05_v9 (405B prompt_v1 assertion, Sept+Dec 2025, multiple passes for full coverage)

Merge step (src/merge/):
  main dataset: s07b tiny 3.5K
  + Dec 2025 405B_prompt_v1 assessment (by hash_id)
  + human annotations (s09)
  + model predictions
  → 3,500 evaluation instances
```

**Two different 405B assessors:**
- `Meta-Llama-3.1-405B`: deprecation assessment, baked into dataset at s05_v8 step
- `Meta-Llama-3.1-405B_prompt_v1`: full re-assessment (assertion), applied via merge step

### s06b — reformat final format

Reformats old s05_v4 LLM assessment output into the TKGU format used by the evaluation pipeline.
Generates hash IDs, renames assessments, assigns TKGU operations (Exists, Add, Mint+Add, Infer, Deprecate).

```bash
export PYTHONPATH="$PWD/src"
python -u -m dataset.emerge.s06b_refactor_final_format \
    --config_file config/dataset/emerge/s06b_refactor_final_format/20250830/config.json
```

### s07 — reduce duplicates (v6)

Two-pass deduplication:
1. Within each file: textual similarity (Jaccard + Levenshtein edit distance)
2. Cross-file: groups instances by their triple sets, keeps the most informative passages

Requires `input_relation_dictionaries` (per-snapshot `documents.jsonl` from s08_extract_relik_index).

```bash
export PYTHONPATH="$PWD/src"
python -u -m dataset.emerge.s07_reduce_duplicates_v6 \
    --config_file config/dataset/emerge/s07_reduce_duplicates/20250910/config.json
```

### s07b — subsample dataset

Stratified subsampling prioritizing: Deprecate → Infer → Mint+Add → Add → remaining.
Output automatically preserves `snapshot_*/` directory structure from input (no manual reorganization needed).

Two configs:
- Full 35K: `config/dataset/emerge/s07b_subsample_dataset/20250910/config.json` (1000/delta, min 400/type)
- Tiny 3.5K: `config/dataset/emerge/s07b_subsample_dataset/20251203_tiny/config.json` (100/delta, min 40/type)

```bash
export PYTHONPATH="$PWD/src"

# Full 35K subsample:
python -u -m dataset.emerge.s07b_subsample_dataset \
    --config_file config/dataset/emerge/s07b_subsample_dataset/20250910/config.json

# Tiny 3.5K for ICML (drawn from the full 35K):
python -u -m dataset.emerge.s07b_subsample_dataset \
    --config_file config/dataset/emerge/s07b_subsample_dataset/20251203_tiny/config.json
```

### s07c — verify duplicates

Verification-only script: checks pairwise similarity within each file and reports similar passages.
Does NOT modify the dataset.

```bash
export PYTHONPATH="$PWD/src"
python -u -m dataset.emerge.s07c_verify_duplicates \
    --config_file config/dataset/emerge/s07c_verify_duplicates/20250901/config.json
```

### Dependencies (pip)

Scripts s07 and s07c require: `scikit-learn`, `scipy`, `python-Levenshtein`
Scripts s07 and s07b require: `nltk` (with `words` corpus and `punkt` tokenizer)

### s05 — LLM assessment of triples (v8, GPU/TGI)

Runs Llama 8B or 405B via TGI (Apptainer) to assess whether triples are supported by text passages.
The v9 sbatch wrapper starts TGI + calls the v8 Python script with `--llm_assessor_name`.

Two assessment modes:
- **8B assertion**: quick first-pass assessment (1 GPU)
- **405B prompt_v1**: high-quality re-assessment for all TKGU types (4 GPUs, GPTQ-INT4)

```bash
# On HPC cluster (sbatch):
sbatch scripts/slurm/dataset/s05_generate_dataset_with_llm_v9_llama405b.sh \
    config/dataset/emerge/s05_generate_dataset_with_llm/20250916_v8_llama_405b_v1/config.json \
    Meta-Llama-3.1-405B_prompt_v1
```

Requires TGI Apptainer image at `.../tgi-docker/text-generation-inference_3.3.4.sif`

### s04 — find interesting snippets (v3, GPU)

Loads full Wikidata KG graph (.pt tensor, ~18GB) to GPU, matches candidate snippets against KG edges.

```bash
sbatch scripts/slurm/dataset/s04_find_interesting_snippets.sh \
    config/dataset/emerge/s04_find_interesting_snippets_v3/20250413_slurm/config.json \
    logs/s04_20250413.log
```

### s03 — textual delta snippets (v8, API-dependent)

Extracts textual context for KG deltas from Wikipedia history dumps. Requires 2 API services running:
1. `s03_API_v6_wiki_mapping.py` — Wikipedia→Wikidata mapping (CPU)
2. `s03_API_v8_only_deltas.py` — delta triple lookup (optional GPU)

```bash
sbatch scripts/slurm/dataset/s03_obtain_textual_delta_snippets_v8.sh \
    config/dataset/emerge/s03_obtain_textual_delta_snippets_v8/20250324/config.json \
    20250324 8000 1 8200 20 logs/s03_20250324.log 36 cuda:0 cuda:1
```

**Note:** Config references Wikipedia dump `20250201` which was deleted. For re-runs, update `wiki_history_directory` to `.../20251101_english/metahistory7zdump/`.

### s08 — extract ReLiK dictionary (API-dependent)

Extracts entity/relation dictionaries for ReLiK benchmark. Uses its own API: `s08_API_v1_wiki_mapping.py`.

```bash
sbatch scripts/slurm/dataset/s08_extract_relik_dictionary.sh \
    config/dataset/emerge/s08_extract_relik_index/20250405_slurm/config.json \
    20250405 8000 1
```

Post-processing: `cat dictionary/<date>/* > dictionary_v2/<date>-wiki_dictionary.jsonl`

### s09 — human annotation and agreement statistics

The annotation pipeline produces human assessments of triple validity and computes
inter-annotator agreement for the paper.

Located in `src/dataset/emerge/`:

| Script | What it does |
|--------|-------------|
| `s09a_subsample_to_annotate_v4.py` | Stratified subsample for annotation (balanced TKGU operations) |
| `s09b_annotate_dataset_v4.py` | Interactive terminal annotation tool (Y/N per triple) |
| `s09c_filter_out_not_annotated.py` | Filter out unannotated instances |
| `s09d_show_and_fix_disagreements.py` | Review and resolve inter-annotator disagreements |
| `s09e_annotation_stats_for_paper.py` | Compute annotation agreement: Cohen's kappa, Fleiss' kappa, Krippendorff's alpha |

Shared utility: `utils/s09_annotate_dataset_utils_v4.py` (agreement metrics, annotation processing).

**Annotation agreement for the paper** (Table in appendix):
```bash
export PYTHONPATH="$PWD/src"
python -u -m dataset.emerge.s09e_annotation_stats_for_paper \
    --config_file config/dataset/emerge/s09e_annotation_stats_for_paper/20250917_revised/config.json
```

This produces Cohen's kappa (H-H, H1-LLM, H2-LLM), Fleiss' kappa, and Krippendorff's alpha
per TKGU operation and overall. Output includes LaTeX table ready for paper insertion.

**Annotation data** (already produced, used by merge step):
`/path/to/storage/wikipedia-processing/output/experiments/s09_annotate_dataset/20250916/solved_disagreements/`

### Phase D — Wikidata dictionary properties

Located in `src/dataset/wikidata/python/`:
- `s04_extract_relik_dictionary_properties.py` — extracts properties from Wikidata history dumps
- `s06_rename_relik_dictionary_properties.py` — reformats JSONL for ReLiK-compatible format

```bash
# Extract properties (sbatch, 64 CPUs, 96GB, 24h):
export PYTHONPATH="$PWD/src"
python -u -m dataset.wikidata.python.s04_extract_relik_dictionary_properties \
    --config_file config/dataset/wikidata/s04_extract_relik_dictionary_properties/20250502/config.json

# Rename/reformat:
python -u -m dataset.wikidata.python.s06_rename_relik_dictionary_properties \
    --config_file config/dataset/wikidata/s06_rename_relik_dictionary_properties/20250509/config.json
```

### Dependencies (pip)

| Scripts | Packages |
|---------|----------|
| s03, s04, s05 | `torch`, `torch_geometric`, `huggingface_hub`, `transformers`, `tiktoken`, `requests`, `fastapi`, `uvicorn`, `psutil`, `py7zr`, `nltk` |
| s07, s07c | `scikit-learn`, `scipy`, `python-Levenshtein` |
| s07, s07b | `nltk` (with `words` corpus and `punkt` tokenizer) |
| s09 | `scikit-learn`, `krippendorff`, `statsmodels`, `pandas`, `numpy` |

---

## Part 3: Quality control of LLM annotations

> **What "garbage" means in this section.** Long LLM annotation runs
> occasionally produce *degenerate* output — repeated tokens, n-gram
> loops, truncation, near-empty responses. We refer to these
> failure-mode outputs as "garbage" throughout this section and in
> the script names (`find_garbage_clusters.py`,
> `build_reinput_for_garbage.py`). The term has no quality judgement
> beyond that — these are the LLM's own off-rails outputs that need
> re-querying, not records we consider useless.

Long LLM annotation runs occasionally produce this kind of off-rails
output, especially when serving via quantised TGI. EMERGE applied an
iterative **detect → reinput → re-run → merge** loop on the
233K-record corpus to catch and re-query these cases. The loop
converged in 3 iterations (871 → 240 → 220 flagged triples on each
pass; final residual rate ≈0.011%).

The four CLI tools in `scripts/dataset/` implement this loop:

| Tool | Step | What it does |
|------|------|--------------|
| `find_garbage_clusters.py` | P3.1 | Detect TGI-degenerate / off-rails LLM output via heuristics (repeated_token, repeated_ngram, low_unique_ratio, repeated_punct, truncation, empty, very_few_words, contradiction). Emits human-readable Markdown + machine-readable JSONL. |
| `inspect_cluster_neighbors.py` | P3.2 | Companion: prints triples adjacent to each detected cluster so you can eyeball whether the cluster boundaries captured the actual degenerate region tightly. |
| `build_reinput_for_garbage.py` | P3.3 | Generic CLI: takes a `--source-root` annotation tree + the cluster JSONL → writes a sparse reinput tree where flagged triples have their 405B-prompt-v1 entries stripped. The output tree is the **input** to a second s05 run that re-queries just those triples. |
| `merge_reinput_into_dataset.py` | P3.5 | Generic CLI: takes the original `--source-root` and the corrected `--reinput-file` (output of the second s05 run on the sparse tree) → splices corrected records back into a NEW merged tree by `hash_id`. Source tree never modified. |

A reviewer running their own LLM annotation pipeline (any LLM, any
infra) can chain these to reach a similar level of QA on their own
output. The example paths in the docstrings are abstract placeholders
(`<your-output-tree>`, `<reinput-tree>`); a runnable end-to-end
example with concrete paths will be added once the rest of the
migration settles.

### P3.1 — Detect garbage clusters

```bash
python scripts/dataset/find_garbage_clusters.py \
    --root <your-output-tree>/llama405b_assessed \
    --output-md output/garbage_clusters_$(date +%Y%m%d_%H%M).md \
    --output-jsonl output/garbage_clusters_$(date +%Y%m%d_%H%M).jsonl
```

Heuristics (each can be tuned via flags; defaults match the EMERGE
release):

- **`repeated_token`** — same token repeated >N times in a row
- **`repeated_ngram`** — same n-gram repeated within a sliding window
- **`low_unique_ratio`** — unique-token ratio below threshold
- **`repeated_punct`** — punctuation-only loops
- **`truncation`** — output ends mid-token
- **`empty`** — empty or whitespace-only output
- **`very_few_words`** — fewer than N words
- **`contradiction`** — assertion + deprecation in the same response

Sliding-window cluster detection groups consecutive flagged triples
(default `--cluster-gap=50`, `--cluster-pad=20`).

### P3.2 — Inspect cluster boundaries (sanity check)

```bash
python scripts/dataset/inspect_cluster_neighbors.py \
    --root <your-output-tree>/llama405b_assessed \
    --clusters-jsonl output/garbage_clusters_<ts>.jsonl \
    --sample-n 5
```

Eyeball the printed neighbours; tighten `--cluster-gap` / `--cluster-pad`
if you see a cluster boundary cutting through a contiguous degenerate
region.

### P3.3 — Build a sparse reinput tree

```bash
python scripts/dataset/build_reinput_for_garbage.py \
    --source-root <your-output-tree>/llama405b_assessed \
    --output-root <reinput-tree>/llama405b_assessed \
    --clusters-jsonl output/garbage_clusters_<ts>.jsonl \
    --output-manifest output/reinput_manifest_$(date +%Y%m%d_%H%M).json
```

Strips the flagged 405B-prompt-v1 annotations from the source records
and writes the resulting sparse tree to `--output-root` (asserted
disjoint from source). This output tree is the input to a re-run of
s05 in single-file mode.

### P3.4 — Re-run s05 on the sparse reinput tree

This step is HPC-bound (it actually invokes the 405B LLM). Use the
existing
[`scripts/slurm/dataset/s05_generate_dataset_with_llm_v9_llama405b.sh`](../../scripts/slurm/dataset/s05_generate_dataset_with_llm_v9_llama405b.sh)
(or your equivalent), pointing it at the reinput tree as input. The
per-triple skip-resume logic in s05 ensures only triples missing a
`Meta-Llama-3.1-405B_prompt_v1` entry get re-queried.

### P3.5 — Merge corrected records back

```bash
python scripts/dataset/merge_reinput_into_dataset.py \
    --source-root <your-output-tree>/llama405b_assessed \
    --reinput-file <reinput-rerun-output>/.../delta_2099-01-01.jsonl \
    --output-root <merged-tree>/llama405b_assessed
```

`hash_id`-keyed splice; whole-record replacement (the reinput record
has the corrected `llm_assessment` plus all original fields). Asserts
`--reinput-file` is outside `--source-root`. Source tree is never
modified.

### Iterate

Run P3.1 again on the merged tree. If new clusters are flagged,
repeat P3.1 → P3.5. For EMERGE the loop converged after 3 iterations
(871 → 240 → 220 flagged triples), reaching a residual rate that we
considered the natural floor for this LLM/infra combination.

---

## Dataset statistics (Jupyter notebooks)

Dataset statistics notebooks are in `src/stats/dataset/`.
See [`src/stats/README.md`](../stats/README.md) for the full list of notebooks
and data dependencies.