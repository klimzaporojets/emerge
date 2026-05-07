# EMERGE Dataset Format

The dataset is organized by Wikidata KG snapshots and weekly deltas:

```
data/
├── evaluation_set/
│   ├── snapshot_2019-01-01/
│   │   ├── delta_2019-01-08.jsonl
│   │   ├── delta_2019-01-15.jsonl
│   │   ├── delta_2019-01-22.jsonl
│   │   ├── delta_2019-01-29.jsonl
│   │   └── delta_2019-02-05.jsonl
│   ├── snapshot_2020-01-01/
│   │   └── ...
│   └── snapshot_2025-01-01/
│       └── ...
└── human_annotation/
    └── solved_disagreements.jsonl
```

Each snapshot directory corresponds to a yearly Wikidata KG snapshot (January 1st).
Each delta file contains 100 instances from a specific week after the snapshot.
Total: 7 snapshots x 5 deltas x 100 instances = 3,500 instances.

## Instance format (JSONL)

Each line in a delta JSONL file is a JSON object with the following structure:

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `hash_id` | string | Unique instance identifier |
| `passage` | string | Wikipedia passage text |
| `mentions` | list | Entity mentions with character offsets and Wikidata QIDs |
| `revision_id` | int | Wikipedia revision ID |
| `revision_date` | string | Wikipedia revision timestamp (ISO 8601) |
| `revision_timestamp` | int | Wikipedia revision time (Unix epoch seconds, paired with `revision_date`) |
| `anchor_title` | string | Wikipedia article title |
| `anchor_page_id` | int | Wikipedia page ID (numeric) |
| `anchor_page_qid` | string | Wikidata QID of the Wikipedia article |
| `paragraph_idx` | int | Paragraph index within the article |
| `delta_dates` | list[2] | `[start_date, end_date]` of the delta period (ISO 8601) |
| `delta_timestamps` | list[2] | `[start, end]` of the delta period (Unix epoch seconds, paired with `delta_dates`) |
| `tkgu_triples` | list | Ground-truth TKGU triples (see below) |
| `predictions` | dict | Model predictions keyed by model name (see below) |

### `mentions` (entity mentions in the passage)

| Field | Type | Description |
|-------|------|-------------|
| `mention_text` | string | Surface form of the mention in the passage |
| `target_entity` | string | Wikipedia article title of the linked entity |
| `start_char` | int | Start character offset in the passage |
| `end_char` | int | End character offset in the passage |
| `qid` | string | Wikidata QID of the entity |

### `tkgu_triples` (ground-truth triples)

Each triple represents a KG update operation derived from the passage:

| Field | Type | Description |
|-------|------|-------------|
| `triple` | list[3] | `[subject_QID, property_PID, object_QID]` — Wikidata identifiers |
| `triple_labels` | list[3] | `[subject_label, property_label, object_label]` — human-readable |
| `tkgu_operations` | list[str] | TKGU operation type(s): `x-triples` (Exists), `e-triples` (Add), `ee-triples` (Mint+Add), `ee-kg-triples` (Infer), `d-triples` (Deprecate) |
| `emerging_head` | bool | Whether the subject entity is new (not in the KG snapshot) |
| `emerging_tail` | bool | Whether the object entity is new |
| `head_creation_date` | string | Date the subject entity was created in Wikidata |
| `head_creation_timestamp` | int | Unix epoch seconds, paired with `head_creation_date` |
| `tail_creation_date` | string | Date the object entity was created in Wikidata |
| `tail_creation_timestamp` | int | Unix epoch seconds, paired with `tail_creation_date` |
| `triple_lifespan_date` | list[2] | `[add_date, delete_date]` — when the triple was added/removed (`null` if still active) |
| `triple_lifespan_timestamp` | list[2] | `[add, delete]` Unix epoch seconds, paired with `triple_lifespan_date` (`null` element if still active) |
| `llm_assessment` | list | LLM verification results (see below) |
| `source_delta_type` | string | How the triple was discovered (e.g., `wikipedia_intersection`) |
| `qualifier_info` | dict \| null | Optional Wikidata qualifier attached to the triple (see below) — present mainly on deprecation triples |

### `llm_assessment` (LLM verification of triple-passage alignment)

| Field | Type | Description |
|-------|------|-------------|
| `llm_name` | string | LLM model used (e.g., `Meta-Llama-3.1-405B_prompt_v1`) |
| `llm_assessment` | bool | Whether the LLM confirms the passage supports this triple |
| `llm_prompt_type` | string | `triple_assertion` or `triple_deprecation` |
| `llm_prompt` | string | LLM's explanation |

### `qualifier_info` (optional Wikidata qualifier on a triple)

A Wikidata qualifier captures temporal or contextual scope on a statement (e.g., an "end time" attached to a `member of sports team` triple). The field is `null` for triples without a qualifier; when present it contains:

| Field | Type | Description |
|-------|------|-------------|
| `qualifier_qid` | string | Wikidata property ID of the qualifier (e.g., `P582` = end time, `P580` = start time) |
| `qualifier_label` | string | Human-readable name of the qualifier property (e.g., `"end time"`) |
| `qualifier_date` | string | Qualifier value as ISO 8601 date |
| `qualifier_timestamp` | int | Qualifier value as Unix epoch seconds (paired with `qualifier_date`) |

### `predictions` (benchmark model outputs)

Keyed by model name. Each model's predictions contain:

| Field | Type | Description |
|-------|------|-------------|
| `predicted_triples` | list | Extracted triples with `action` and `extracted_relation` |
| `predicted_triples_oie` | list | Open IE triples (free-text, not linked to Wikidata) |
| `predicted_triples_entities_to_kg` | list | Triples linking new entities to existing KG entities |
| `model` | string | Model identifier |
| `model_type` | string | Model family (e.g., `edc`, `kg-gen`, `rakg`, `rebel`, `relik`) |

## Annotation data

`data/human_annotation/solved_disagreements.jsonl` contains human annotation data used for
inter-annotator agreement statistics (Cohen's kappa, Fleiss' kappa, Krippendorff's alpha).
