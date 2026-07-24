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
    """Retriever #1 — reuse ReLiK's OWN per-snapshot index (pre-built, `--indices`).

    One `model([passage])` call yields both entity and relation candidates; we read
    `RelikOutput.candidates.span` (entities; metadata['wikidata']=QID) and
    `.candidates.triplet` (relations; metadata['property']=PID) -- exactly the
    retriever output, ranked. Result cached per passage so the two `retrieve_*` calls
    reuse one forward pass. (Mirrors src/benchmarks/wrappers/relik/relik_benchmark_v2.py.)
    """

    def __init__(
        self,
        config_name: str,
        entity_index_path: str,
        relation_index_path: str,
        precision: str = "16",
        device: str = "cpu",
        cache_dir: str | None = None,
    ) -> None:
        from relik import Relik
        idx = "relik.retriever.indexers.inmemory.InMemoryDocumentIndex.from_pretrained"
        index_param = {
            "span": {"_target_": idx, "name_or_path": entity_index_path, "precision": precision},
            "triplet": {"_target_": idx, "name_or_path": relation_index_path, "precision": precision},
        }
        self.model = Relik.from_pretrained(
            config_name, device=device, index=index_param, cache_dir=cache_dir
        )
        self._cache: dict[str, object] = {}

    def _run(self, passage: str):
        if passage not in self._cache:
            self._cache[passage] = self.model([passage])[0]
        return self._cache[passage]

    @staticmethod
    def _flatten(nested, meta_key: str) -> list[tuple[str, str]]:
        """ReLiK candidates are nested (windows x candidates, sometimes one level
        deeper). Flatten -> [(id, label)] deduped, order (= ranking) preserved."""
        seen: set[str] = set()
        out: list[tuple[str, str]] = []

        def add(c) -> None:
            cid = c.metadata.get(meta_key)
            if cid and cid not in seen:
                seen.add(cid)
                out.append((cid, c.text))

        for group in nested:
            for c in group:
                if isinstance(c, list):
                    for c2 in c:
                        add(c2)
                else:
                    add(c)
        return out

    def retrieve_entities(self, passage: str, top_k: int) -> list[Hit]:
        pairs = self._flatten(self._run(passage).candidates.span, "wikidata")
        return [Hit(cid, lbl, 1.0 / (i + 1)) for i, (cid, lbl) in enumerate(pairs[:top_k])]

    def retrieve_relations(self, passage: str, top_k: int) -> list[Hit]:
        pairs = self._flatten(self._run(passage).candidates.triplet, "property")
        return [Hit(cid, lbl, 1.0 / (i + 1)) for i, (cid, lbl) in enumerate(pairs[:top_k])]
