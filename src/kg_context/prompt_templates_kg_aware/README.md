# KG-aware prompt templates

On-disk prompt templates for the **KG-conditioned baseline** (Task 1). `compose_kg_prompt.py` loads the
template for each point, fills the `{placeholders}`, and writes the finished prompt to the eval-set's
`prompt` field (the original `passage` stays pristine). A dedicated runner then sends `prompt` to the LLM
**verbatim** — unlike KGGen / EDC+, no baseline builds its own prompt around the text.

## Files
- `unified.txt` — one prompt per passage, all five TKGU ops together (op label `all`).
- `exists.txt` / `add.txt` / `mint_add.txt` / `infer.txt` / `deprecate.txt` — one prompt per op (per_op mode).
- `output_format.txt` — the shared machine-parseable output spec (filled into `{output_format}`).

## Placeholders (filled by `augment.template_fields` → `augment.fill_template`)
- `{passage}` — the pristine original passage.
- `{relation_types}` — "Candidate relation types:" + bullet list `label (Pid)` (empty if `include_relations` off).
- `{entities}` — "Candidate existing KG entities:" + bullet list `label (Qid)` (empty if `include_entities` off).
- `{existing_triples}` — "Existing KG triples …" + bullets `(h (Qid), r (Pid), t (Qid))` (empty if none).
- `{kg_context}` — the three above bundled under one `##KG CONTEXT##` header (use instead of the granular three).
- `{output_format}` — contents of `output_format.txt`.

An empty placeholder makes its whole `##SECTION##` disappear (`fill_template` tidies empty sections), so
`text_only` naturally drops `##KG CONTEXT##`. If a template references a placeholder the op's field builder
does **not** provide, `fill_template` **hard-fails** (a per-op field need can't silently render empty).

## Per-op field logic (the extensible seam)
Every op currently uses the generic field builder (`augment._default_fields`). When an op needs **different**
fields/logic, register a builder in `augment.OP_FIELD_BUILDERS[op_label]` and add the matching placeholder to
that op's template. **TODO (to analyze):** Mint+Add and Infer likely want their own logic (e.g. Infer: emphasise
the not-in-text KG entity, or add relation-pattern demos `{pattern_examples}`); any such change may also be
reflected in `unified.txt`. Op labels: `exists, add, mint_add, infer, deprecate, all`.
