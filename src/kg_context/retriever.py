"""RETRIEVE: index-wide dense retrieval of candidate entities / relations.

CORRECTED DESIGN (2026-07-24): a NORMAL dense retriever. `retrieve(passage, k)` returns
the top-k entities (or relations) from the WHOLE snapshot vocabulary by passage
nearest-neighbor. No anchor restriction, no traversal, no k-hop (k-hop is the more
advanced model = future work).

Two interchangeable backends, same `retrieve()` interface:
  - MpnetIndexRetriever : queries an all-mpnet index built by `index_builder.py` (retriever #2).
  - ReLiK's retriever   : reuse ReLiK's per-snapshot index (retriever #1) -- TO WIRE after
                          a read-only recon of ReLiK's standalone retrieve API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Hit:
    id: str          # QID or PID
    label: str | None
    score: float


class MpnetIndexRetriever:
    """Index-wide retrieval over a prebuilt all-mpnet index (from `index_builder.py`).

    `retrieve(passage, k)` -> top-k Hits from the whole vocabulary. Loads the emb matrix
    as a memmap (never fully in RAM). Brute-force dot product per query; swap in FAISS if
    we need speed on the full (millions-of-rows) index. Pass a shared `model` so the
    entity and relation retrievers don't each load their own encoder.
    """

    def __init__(
        self,
        index_dir: str,
        model=None,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
    ) -> None:
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device=device)
        self.model = model
        self.emb = np.load(os.path.join(index_dir, "emb.npy"), mmap_mode="r")  # [N, d]
        with open(os.path.join(index_dir, "ids.txt"), encoding="utf-8") as f:
            self.ids = f.read().splitlines()
        with open(os.path.join(index_dir, "labels.txt"), encoding="utf-8") as f:
            self.labels = f.read().splitlines()

    def retrieve(self, query_text: str, top_k: int) -> list[Hit]:
        q = self.model.encode(
            [query_text], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        sims = self.emb @ q                          # (N,), cosine (both normalized)
        k = min(top_k, sims.shape[0])
        if k <= 0:
            return []
        part = np.argpartition(-sims, k - 1)[:k]      # top-k unsorted
        order = part[np.argsort(-sims[part])]         # then sort those k
        return [Hit(self.ids[i], self.labels[i], float(sims[i])) for i in order]

    @staticmethod
    def shared_model(model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cpu"):
        """Build one encoder to share across the entity + relation retrievers."""
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, device=device)


# ---------------------------------------------------------------------------
# Unified candidate-retriever interface: retrieve_entities / retrieve_relations.
# Two backends, both index-wide, no traversal. The runner uses this interface so
# the config can pick "mpnet_index" or "relik" with no other change.
# ---------------------------------------------------------------------------

class MpnetCandidateRetriever:
    """Retriever #2 — our all-mpnet index (built by index_builder.py). Two indexes
    (entities, relations) sharing one encoder."""

    def __init__(
        self,
        entities_index_dir: str,
        relations_index_dir: str,
        model=None,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
    ) -> None:
        if model is None:
            model = MpnetIndexRetriever.shared_model(model_name, device)
        self.ent = MpnetIndexRetriever(entities_index_dir, model=model)
        self.rel = MpnetIndexRetriever(relations_index_dir, model=model)

    def retrieve_entities(self, passage: str, top_k: int) -> list[Hit]:
        return self.ent.retrieve(passage, top_k)

    def retrieve_relations(self, passage: str, top_k: int) -> list[Hit]:
        return self.rel.retrieve(passage, top_k)


class RelikCandidateRetriever:
    """Retriever #1 — ReLiK's RETRIEVER ONLY (reader skipped), over ReLiK's pre-built per-snapshot index.

    We only want retrieved CANDIDATES, not ReLiK's extracted triplets, so we load with `reader=None`
    (skips the deberta-v3-large reader: no ~1.7 GB load, no per-passage reader forward pass -> big CPU
    speedup, and no spacy windowization) and call the underlying `GoldenRetriever` for each task
    DIRECTLY on the passage. This gives REAL cosine scores (`RetrievedSample.score`), unlike the
    reader pipeline which drops them.

    `model._retriever` = {TaskType.SPAN: entity retriever (-> entity index), TaskType.TRIPLET: relation
    retriever (-> relation index)}. `GoldenRetriever.retrieve([text], k) -> [[RetrievedSample]]`;
    each sample has `.score` and `.document` (`.text` = label, `.metadata['wikidata'|'property']` = QID/PID).
    Retrieval is at PASSAGE granularity (median eval passage ~71 words; the encoder truncates the long
    tail ~>512 tokens -- window-based retrieval is a possible later refinement).
    """

    def __init__(
        self,
        config_name: str,
        entity_index_path: str,
        relation_index_path: str,
        precision: str = "16",
        device: str = "cpu",
        cache_dir: str | None = None,
        entity_config_name: str | None = None,
    ) -> None:
        """`entity_config_name` selects the ENTITY (span) retriever model, enabling the index ablation:
          - None / == config_name  -> NATIVE: one model (e.g. relik-cie-large) serves BOTH span+triplet
            from a SINGLE load; entity index = its own e5-small-v2-matryoshka (384-dim).
          - != config_name (e.g. `sapienzanlp/relik-entity-linking-large`) -> SPLIT: entity retriever is
            loaded from THAT model (e.g. e5-base-v2 AIDA, 768-dim) over `entity_index_path`, while the
            RELATION retriever stays native (`config_name` over `relation_index_path`). Only the entity
            side changes -> a clean native-vs-AIDA ablation. See rebuttal_baseline.md TODO."""
        import time
        from relik import Relik
        from relik.inference.data.objects import TaskType
        idx = "relik.retriever.indexers.inmemory.InMemoryDocumentIndex.from_pretrained"

        def _span_idx(path):
            return {"span": {"_target_": idx, "name_or_path": path, "precision": precision}}

        def _trip_idx(path):
            return {"triplet": {"_target_": idx, "name_or_path": path, "precision": precision}}

        entity_config_name = entity_config_name or config_name
        # reader=None -> ONLY retriever encoders + in-memory indexes load (no deberta reader/forward).
        t0 = time.time()
        if entity_config_name == config_name:
            # NATIVE: single model, both indexes in one load.
            print(f"[ReLiK] loading RETRIEVER-ONLY (reader=None), model={config_name}: entity={entity_index_path}, "
                  f"relation={relation_index_path}; ~15 GB entity -> minutes, watch `free -g` ...", flush=True)
            model = Relik.from_pretrained(
                config_name, device=device,
                index={**_span_idx(entity_index_path), **_trip_idx(relation_index_path)},
                reader=None, cache_dir=cache_dir)
            self._span = model._retriever[TaskType.SPAN]
            self._rel = model._retriever[TaskType.TRIPLET]
        else:
            # SPLIT (AIDA ablation): entity model != relation model -> two separate reader-less loads.
            # Relations stay native (config_name over relation_index_path) so only the entity side changes.
            print(f"[ReLiK] loading SPLIT retriever (reader=None): ENTITY model={entity_config_name} "
                  f"(index={entity_index_path}, ~2x bigger for e5-base/768-dim), RELATION model={config_name} "
                  f"(index={relation_index_path}); watch `free -g` ...", flush=True)
            ent_model = Relik.from_pretrained(
                entity_config_name, device=device, index=_span_idx(entity_index_path),
                reader=None, cache_dir=cache_dir)
            # `config_name` (relik-cie-large) is a task=BOTH model, so relik's load_index REQUIRES a SPAN
            # index too (task drives it, and BOTH loads span+triplet) even though we only use TRIPLET here.
            # Passing triplet-only -> `KeyError: TaskType.SPAN`. Point the SPAN slot at the RELATION index as
            # a harmless, never-queried placeholder (valid dir, ~18 MB) instead of loading the 10 GB native
            # entity index we don't need. We then take ONLY _retriever[TaskType.TRIPLET] (the real relation
            # retriever, linked to relation_index_path). Verified against relik utils.load_index (task=BOTH).
            rel_model = Relik.from_pretrained(
                config_name, device=device,
                index={**_span_idx(relation_index_path), **_trip_idx(relation_index_path)},
                reader=None, cache_dir=cache_dir)
            self._span = ent_model._retriever[TaskType.SPAN]
            self._rel = rel_model._retriever[TaskType.TRIPLET]
        print(f"[ReLiK] retriever + indexes loaded in {time.time() - t0:.0f}s; ready to retrieve.", flush=True)

    @staticmethod
    def _hits(samples, meta_key: str, top_k: int) -> list[Hit]:
        """RetrievedSample list (score-ranked) -> deduped Hits, real cosine scores, order preserved."""
        seen: set[str] = set()
        out: list[Hit] = []
        for s in samples:
            cid = (s.document.metadata or {}).get(meta_key)
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(Hit(cid, s.document.text, float(s.score)))
            if len(out) >= top_k:
                break
        return out

    # NOTE: batch_size MUST be set. GoldenRetriever.retrieve defaults batch_size=None, which makes the
    # DataLoader skip automatic batching -> default_collate_fn gets a single (text, text_pair) sample
    # instead of a list and crashes on the None text_pair (`'NoneType' object is not subscriptable`).
    # ReLiK's own __call__ always passes batch_size; the standalone retriever path (this one) must too.
    # num_workers=0 -> no DataLoader subprocess (one passage per call; avoids worker-process overhead).
    def retrieve_entities(self, passage: str, top_k: int) -> list[Hit]:
        out = self._span.retrieve([passage], k=top_k, batch_size=1, num_workers=0)
        return self._hits(out[0] if out else [], "wikidata", top_k)

    def retrieve_relations(self, passage: str, top_k: int) -> list[Hit]:
        out = self._rel.retrieve([passage], k=top_k, batch_size=1, num_workers=0)
        return self._hits(out[0] if out else [], "property", top_k)
