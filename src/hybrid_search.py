"""
Hybrid Search: combines BM25 (keyword) search with vector (semantic) search
using Reciprocal Rank Fusion (RRF), then re-ranks the fused candidates with
a cross-encoder for higher precision. Everything runs locally - free.
"""

import re

from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.config import EMBEDDING_MODEL, CHROMA_DB_PATH, TOP_K, RRF_K


def tokenize(text: str) -> list[str]:
    """
    Lowercase + strip punctuation before splitting into words, so that
    'GDP)' and 'GDP' (or '"Make' and 'Make') are treated as the SAME token
    for keyword matching, instead of being seen as different words just
    because of attached quotes/brackets/periods/commas.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # replace punctuation with a space
    return text.split()


class HybridSearchEngine:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        # Small, free, local cross-encoder used ONLY to re-rank the top
        # candidates after RRF fusion - it does not replace BM25/vector search.
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection("documents")

        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_ids = []
        self._rebuild_bm25()

    # ---------------------------------------------------------------
    # Indexing
    # ---------------------------------------------------------------
    def add_documents(self, chunks, source: str):
        if not chunks:
            return
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        embeddings = self.embedder.encode(chunks).tolist()
        self.collection.add(
            documents=chunks, embeddings=embeddings, ids=ids,
            metadatas=[{"source": source} for _ in chunks],
        )
        self._rebuild_bm25()

    def clear_all(self):
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_ids = []

    def get_all_sources(self):
        all_data = self.collection.get()
        sources = set()
        for meta in all_data.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)

    def _rebuild_bm25(self):
        all_data = self.collection.get()
        self.bm25_corpus = all_data["documents"]
        self.bm25_ids = all_data["ids"]
        tokenized = [tokenize(doc) for doc in self.bm25_corpus]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    # ---------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------

    def _vector_search(self, query: str, top_k: int):
        if self.collection.count() == 0:
            return []
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=top_k,
            include=["documents", "distances"],
        )
        ids = results["ids"][0] if results["ids"] else []
        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        # Chroma returns a distance (lower = closer); convert to a similarity
        # score (higher = better) so it's on the same "higher is better" scale
        # as BM25/RRF/rerank for easy side-by-side comparison.
        similarities = [round(1 - d, 4) for d in distances]
        return list(zip(ids, docs, similarities))

    def _bm25_search(self, query: str, top_k: int):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.bm25_ids, self.bm25_corpus, scores), key=lambda x: x[2], reverse=True)
        return [(i, d, round(float(s), 4)) for i, d, s in ranked[:top_k]]

    def hybrid_search(self, query: str, top_k: int = TOP_K):
        """
        Returns each result with ALL of its scores at every stage, so you
        can see exactly how BM25 + vector search + RRF fusion + re-ranking
        each contributed to the final ordering:
          - bm25_score: raw keyword-match score (0 if not found by BM25)
          - vector_score: semantic similarity score (0 if not found by vector search)
          - rrf_score: the fused rank-based score combining both - RRF score = 1/(k + rank)
          - rerank_score: the cross-encoder's final precision re-score
        """
        vector_results = self._vector_search(query, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)

        rrf_scores, texts = {}, {}
        bm25_score_map, vector_score_map = {}, {}

        for rank, (doc_id, text, sim) in enumerate(vector_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)
            texts[doc_id] = text
            vector_score_map[doc_id] = sim

        for rank, (doc_id, text, score) in enumerate(bm25_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)
            texts[doc_id] = text
            bm25_score_map[doc_id] = score

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = fused[:top_k * 2]

        if not candidates:
            return []

        pairs = [[query, texts[doc_id]] for doc_id, _ in candidates]
        rerank_scores = self.reranker.predict(pairs)

        combined = list(zip(candidates, rerank_scores))
        combined.sort(key=lambda x: x[1], reverse=True)

        results = []
        for (doc_id, rrf_score), rerank_score in combined[:top_k]:
            results.append({
                "text": texts[doc_id],
                "bm25_score": bm25_score_map.get(doc_id, 0.0),
                "vector_score": vector_score_map.get(doc_id, 0.0),
                "rrf_score": round(rrf_score, 4),
                "rerank_score": round(float(rerank_score), 4),
            })
        return results