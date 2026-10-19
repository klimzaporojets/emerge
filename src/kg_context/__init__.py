"""KG-context augmentation for the EMERGE benchmark (diagnostic probe).

Given an evaluation instance, retrieve KG context (candidate entities + relation
types) for its anchor entity from a KG snapshot, and append a `KG context` block
to the passage. The existing KGGen / EDC+ baseline wrappers then run UNCHANGED on
the augmented input (input-augmentation only — no baseline or scorer edits).

Components. "index" = dense EMBEDDING
matrices (fuzzy retrieval) only; the exact snapshot edge-check is a KG-LOOKUP, not an index:
  index_builder — build the all-mpnet entity/relation embedding INDEX (retriever #2)
  retriever     — MpnetIndexRetriever: passage -> top-k candidates, index-wide (CPU)
  kg_lookup     — KGLookup: exact has_edge(anchor,rel,ent) membership (EXISTS/DEPRECATE)
  augment       — build the KG-context block -> augmented eval-set JSONL

Conditions: text-only (have) | KG-RAG (retriever top-k) | oracle (gold guaranteed,
with an @{0,2,4,8,16,32} distractor sweep).
"""
