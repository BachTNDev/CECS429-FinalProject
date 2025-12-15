# search/hybrid.py - FIXED: Handles inverted Dense scores
from typing import Dict, List, Tuple
from collections import defaultdict

def _min_max_normalize(scores: Dict[str, float], invert: bool = False) -> Dict[str, float]:
    """
    Min-max normalize scores to [0, 1] range
    
    Args:
        scores: Dict of doc_id -> score
        invert: If True, invert scores (for distance metrics where lower=better)
    """
    if not scores:
        return {}
    
    vals = list(scores.values())
    min_s, max_s = min(vals), max(vals)
    
    # Handle edge case: all scores are the same
    if max_s == min_s:
        return {doc_id: 1.0 for doc_id in scores}
    
    if invert:
        # For distance metrics: lower is better, so invert
        # After inversion, higher normalized score = better
        return {doc_id: (max_s - s) / (max_s - min_s) for doc_id, s in scores.items()}
    else:
        # For similarity metrics: higher is better
        return {doc_id: (s - min_s) / (max_s - min_s) for doc_id, s in scores.items()}


def _rank_normalize(results: List[Tuple[str, float]]) -> Dict[str, float]:
    """
    Reciprocal rank normalization (score-scale independent)
    Score = 1 / (rank + 60)
    """
    scores = {}
    for rank, (doc_id, _) in enumerate(results):
        scores[doc_id] = 1.0 / (rank + 60)
    return scores


def hybrid_fusion(
    bm25_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    alpha: float = 0.5,
    k: int = 10,
    method: str = "rrf"  # Default to RRF (more robust)
) -> List[Tuple[str, float]]:
    """
    Hybrid fusion of BM25 and Dense results
    
    Args:
        bm25_results: List of (doc_id, score) from BM25 (higher=better)
        dense_results: List of (doc_id, distance) from Dense (lower=better for cosine distance)
        alpha: Weight for BM25 (0.0 = pure Dense, 1.0 = pure BM25)
        k: Number of results to return
        method: "rrf" (recommended) or "min_max"
    
    Returns:
        List of (doc_id, fused_score) sorted by score
    """
    
    if method == "rrf":
        # Reciprocal Rank Fusion - rank-based, scale-independent
        bm25_norm = _rank_normalize(bm25_results)
        dense_norm = _rank_normalize(dense_results)
    else:
        # Min-max normalization with proper handling of distance metrics
        bm25_scores = {d: s for d, s in bm25_results}
        dense_scores = {d: s for d, s in dense_results}
        
        # BM25: higher is better (similarity)
        bm25_norm = _min_max_normalize(bm25_scores, invert=False)
        
        # Dense FAISS: LOWER is better (distance), so INVERT
        dense_norm = _min_max_normalize(dense_scores, invert=True)
    
    # Combine scores
    fused_scores = defaultdict(float)
    all_docs = set(bm25_norm.keys()) | set(dense_norm.keys())
    
    for doc_id in all_docs:
        s_bm25 = bm25_norm.get(doc_id, 0.0)
        s_dense = dense_norm.get(doc_id, 0.0)
        fused_scores[doc_id] = alpha * s_bm25 + (1.0 - alpha) * s_dense
    
    # Sort and return top k
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    k: int = 10,
    rrf_k: int = 60
) -> List[Tuple[str, float]]:
    """
    Pure Reciprocal Rank Fusion (parameter-free, robust)
    
    RRF score = sum over all methods of: 1 / (rank + k)
    
    This is rank-based and doesn't care about score scales.
    """
    scores = defaultdict(float)
    
    # Add BM25 ranks
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] += 1.0 / (rank + rrf_k)
    
    # Add Dense ranks
    for rank, (doc_id, _) in enumerate(dense_results):
        scores[doc_id] += 1.0 / (rank + rrf_k)
    
    # Sort and return
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]