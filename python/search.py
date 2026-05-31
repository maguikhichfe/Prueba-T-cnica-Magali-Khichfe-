"""
search.py — Parte 3 y 5: Recuperación semántica/léxica.
Soporta tanto índice FAISS (sentence-transformers) como TF-IDF (fallback).
"""
import pickle
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

META_PATH  = Path(__file__).parent / "index_meta.pkl"
TOP_K      = 4
MIN_SCORE  = 0.30


class SemanticSearch:
    def __init__(self):
        if not META_PATH.exists():
            raise FileNotFoundError(
                f"Índice no encontrado en {META_PATH}. "
                "Ejecuta 'python ingest.py' primero."
            )
        log.info("Cargando índice...")
        with open(META_PATH, "rb") as f:
            self.meta = pickle.load(f)
        self.mode   = self.meta["mode"]
        self.chunks = self.meta["chunks"]

        if self.mode == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            import faiss as _faiss
            self.faiss  = _faiss
            self.model  = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            self.index  = self.meta["faiss_index"]
        else:
            self.vectorizer = self.meta["vectorizer"]
            self.vectors    = self.meta["vectors"]   # shape (N, dim)

        log.info(f"Índice listo ({self.mode}, {len(self.chunks)} fragmentos).")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        if not query.strip():
            return []

        if self.mode == "sentence-transformers":
            return self._search_faiss(query, top_k)
        else:
            return self._search_tfidf(query, top_k)

    def _search_faiss(self, query, top_k):
        vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        import faiss
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < MIN_SCORE:
                continue
            c = self.chunks[idx]
            results.append({"text": c["text"], "source": c["source"], "score": float(score)})
        return results

    def _search_tfidf(self, query, top_k):
        q_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        norm = np.linalg.norm(q_vec)
        if norm == 0:
            return []
        q_vec /= norm
        scores = (self.vectors @ q_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < MIN_SCORE:
                continue
            c = self.chunks[idx]
            results.append({"text": c["text"], "source": c["source"], "score": score})
        return results


_searcher = None

def get_searcher() -> SemanticSearch:
    global _searcher
    if _searcher is None:
        _searcher = SemanticSearch()
    return _searcher
