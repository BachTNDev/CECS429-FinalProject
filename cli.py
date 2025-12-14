# cli.py
import argparse
import pickle
from pathlib import Path

from search.io_utils import load_corpus, load_queries, load_qrels
from search.bm25 import BM25Index
from search.dense_faiss import DenseFaissIndex
from search.hybrid import hybrid_fusion
from search.eval import evaluate_system

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def cmd_build(args):
    corpus = load_corpus(args.corpus)

    print("Building BM25 index ...")
    bm25 = BM25Index()
    bm25.build_from_corpus(corpus)
    with open(MODELS_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    print("Building Dense FAISS index ...")
    dense = DenseFaissIndex()
    dense.build_from_corpus(corpus)
    with open(MODELS_DIR / "dense.pkl", "wb") as f:
        pickle.dump(dense, f)

    print("Done.")

def load_models():
    with open(MODELS_DIR / "bm25.pkl", "rb") as f:
        bm25: BM25Index = pickle.load(f)
    with open(MODELS_DIR / "dense.pkl", "rb") as f:
        dense: DenseFaissIndex = pickle.load(f)
    return bm25, dense

def cmd_search(args):
    bm25, dense = load_models()
    query = args.query
    k = args.k

    bm25_res = bm25.search(query, k)
    dense_res = dense.search(query, k)
    hybrid_res = hybrid_fusion(bm25_res, dense_res, alpha=args.alpha, k=k)

    print("\n=== BM25 ===")
    for rank, (doc_id, score) in enumerate(bm25_res, start=1):
        print(f"{rank:2d}. {doc_id}  {score:.4f}")

    print("\n=== Dense ===")
    for rank, (doc_id, score) in enumerate(dense_res, start=1):
        print(f"{rank:2d}. {doc_id}  {score:.4f}")

    print("\n=== Hybrid ===")
    for rank, (doc_id, score) in enumerate(hybrid_res, start=1):
        print(f"{rank:2d}. {doc_id}  {score:.4f}")

def cmd_eval(args):
    bm25, dense = load_models()
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)

    print("Evaluating BM25 ...")
    bm25_metrics = evaluate_system(lambda q, k=100: bm25.search(q, k), queries, qrels)
    print(bm25_metrics)

    print("Evaluating Dense ...")
    dense_metrics = evaluate_system(lambda q, k=100: dense.search(q, k), queries, qrels)
    print(dense_metrics)

    print("Evaluating Hybrid ...")
    def hybrid_run(q: str, k: int = 100):
        bm25_res = bm25.search(q, k)
        dense_res = dense.search(q, k)
        return hybrid_fusion(bm25_res, dense_res, alpha=args.alpha, k=k)

    hybrid_metrics = evaluate_system(hybrid_run, queries, qrels)
    print(hybrid_metrics)

def main():
    parser = argparse.ArgumentParser(description="Hybrid search engine CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_build = subparsers.add_parser("build")
    p_build.add_argument("--corpus", type=str, default=str(DATA_DIR / "corpus.jsonl"))
    p_build.set_defaults(func=cmd_build)

    p_search = subparsers.add_parser("search")
    p_search.add_argument("query", type=str)
    p_search.add_argument("-k", type=int, default=10)
    p_search.add_argument("--alpha", type=float, default=0.5)
    p_search.set_defaults(func=cmd_search)

    p_eval = subparsers.add_parser("eval")
    p_eval.add_argument("--queries", type=str, default=str(DATA_DIR / "queries.tsv"))
    p_eval.add_argument("--qrels", type=str, default=str(DATA_DIR / "qrels.tsv"))
    p_eval.add_argument("--alpha", type=float, default=0.5)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
