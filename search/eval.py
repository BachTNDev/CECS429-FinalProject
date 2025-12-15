# search/eval.py
import math
from typing import Dict, List, Tuple, Set

def ndcg_at_k(ranked_docs: List[str], rel_docs: Set[str], k: int = 10) -> float:
    gains = [1.0 if d in rel_docs else 0.0 for d in ranked_docs[:k]]

    def dcg(scores: List[float]) -> float:
        return sum(score / math.log2(i + 2) for i, score in enumerate(scores))

    dcg_val = dcg(gains)
    ideal_gains = sorted(gains, reverse=True)
    idcg_val = dcg(ideal_gains)
    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val

def mrr_at_k(ranked_docs: List[str], rel_docs: Set[str], k: int = 10) -> float:
    for i, d in enumerate(ranked_docs[:k]):
        if d in rel_docs:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(ranked_docs: List[str], rel_docs: Set[str], k: int = 100) -> float:
    if not rel_docs:
        return 0.0
    retrieved = set(ranked_docs[:k])
    hit = len(retrieved & rel_docs)
    return hit / len(rel_docs)

def evaluate_system(
    run_func,
    queries: Dict[str, str],
    qrels: Dict[str, Set[str]],
    k_ndcg: int = 10,
    k_mrr: int = 10,
    k_recall: int = 100,
    progress_every: int = 500,  # NEW
) -> Dict[str, float]:
    ndcgs, mrrs, recalls = [], [], []
    for i, (qid, qtext) in enumerate(queries.items(), start=1):
        rel_docs = qrels.get(qid, set())
        if not rel_docs:
            continue

        results = run_func(qtext, max(k_ndcg, k_mrr, k_recall))
        ranked_docs = [doc_id for doc_id, _ in results]

        ndcgs.append(ndcg_at_k(ranked_docs, rel_docs, k_ndcg))
        mrrs.append(mrr_at_k(ranked_docs, rel_docs, k_mrr))
        recalls.append(recall_at_k(ranked_docs, rel_docs, k_recall))

        if progress_every and i % progress_every == 0:
            print(f"  processed {i} queries...")

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        f"nDCG@{k_ndcg}": mean(ndcgs),
        f"MRR@{k_mrr}": mean(mrrs),
        f"Recall@{k_recall}": mean(recalls),
    }
