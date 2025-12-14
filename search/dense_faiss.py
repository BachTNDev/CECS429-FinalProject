# search/dense_faiss.py
from typing import Dict, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class DenseFaissIndex:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hnsw_m: int = 32
    ):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        # HNSW index in cosine space (we will store normalized vectors).
        self.index = faiss.IndexHNSWFlat(self.dim, hnsw_m)
        self.index.hnsw.efSearch = 64
        self.index.hnsw.efConstruction = 200

        self.doc_ids: List[str] = []
        self.docs: Dict[str, str] = {}

    def _encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype("float32")

    def build_from_corpus(self, corpus: Dict[str, str], batch_size: int = 64):
        doc_ids = list(corpus.keys())
        texts = [corpus[doc_id] for doc_id in doc_ids]

        self.doc_ids = []
        self.docs = corpus

        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]
            batch_ids = doc_ids[start:end]
            emb = self._encode(batch_texts)
            self.index.add(emb)
            self.doc_ids.extend(batch_ids)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        q_emb = self._encode([query])
        D, I = self.index.search(q_emb, k)
        scores_ids: List[Tuple[str, float]] = []

        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            doc_id = self.doc_ids[idx]
            scores_ids.append((doc_id, float(score)))

        return scores_ids

    def get_document(self, doc_id: str) -> str:
        return self.docs.get(doc_id, "")
