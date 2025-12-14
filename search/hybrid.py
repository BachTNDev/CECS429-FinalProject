# search/hybrid.py
from typing import Dict, List, Tuple
from collections import defaultdict

def _min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    min_s, max_s = min(vals), max(vals)
    if max_s == min_s:
        # avoid divide-by-zero; all scores become 1.0
        return {doc_id: 1.0 for doc_id in scores}
    return {doc_id: (s - min_s) / (max_s - min_s) for doc_id, s in scores.items()}

def hybrid_fusion(
    bm25_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    alpha: float = 0.5,
    k: int = 10,
) -> List[Tuple[str, float]]:
    """
    alpha = weight for BM25; (1 - alpha) = weight for dense.
    Input lists are [(doc_id, score)], already sorted by each method.
    """
    bm25_scores = {d: s for d, s in bm25_results}
    dense_scores = {d: s for d, s in dense_results}

    bm25_norm = _min_max_normalize(bm25_scores)
    dense_norm = _min_max_normalize(dense_scores)

    fused_scores = defaultdict(float)

    all_docs = set(bm25_norm.keys()) | set(dense_norm.keys())
    for doc_id in all_docs:
        s_bm25 = bm25_norm.get(doc_id, 0.0)
        s_dense = dense_norm.get(doc_id, 0.0)
        fused_scores[doc_id] = alpha * s_bm25 + (1.0 - alpha) * s_dense

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]
