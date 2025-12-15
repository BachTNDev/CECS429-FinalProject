# cli.py - Memory-optimized version
import argparse
import pickle
import gc
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
    """Build indexes with memory optimization"""
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} documents")
    
    # Build BM25 first (lighter)
    if not args.dense_only:
        print("\n" + "="*60)
        print("Building BM25 index ...")
        print("="*60)
        bm25 = BM25Index()
        bm25.build_from_corpus(corpus)
        
        bm25_path = MODELS_DIR / "bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)
        print(f"✅ BM25 saved to {bm25_path}")
        
        # Free memory
        del bm25
        gc.collect()
    
    # Build Dense second (heavier)
    if not args.bm25_only:
        print("\n" + "="*60)
        print("Building Dense FAISS index ...")
        print("="*60)
        
        # Reduce batch size for memory efficiency
        batch_size = args.batch_size if args.batch_size else 32
        print(f"Using batch_size={batch_size}")
        
        dense = DenseFaissIndex()
        dense.build_from_corpus(corpus, batch_size=batch_size)
        
        dense_path = MODELS_DIR / "dense.pkl"
        with open(dense_path, "wb") as f:
            pickle.dump(dense, f)
        print(f"✅ Dense saved to {dense_path}")
        
        # Free memory
        del dense
        gc.collect()
    
    print("\n✅ All done!")

def load_models():
    with open(MODELS_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(MODELS_DIR / "dense.pkl", "rb") as f:
        dense = pickle.load(f)
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

    p_build = subparsers.add_parser("build", help="Build search indexes")
    p_build.add_argument("--corpus", type=str, default=str(DATA_DIR / "corpus.jsonl"))
    p_build.add_argument("--batch-size", type=int, default=32, 
                        help="Batch size for encoding (lower = less memory)")
    p_build.add_argument("--bm25-only", action="store_true", 
                        help="Build only BM25 index")
    p_build.add_argument("--dense-only", action="store_true", 
                        help="Build only Dense index")
    p_build.set_defaults(func=cmd_build)

    p_search = subparsers.add_parser("search", help="Search the index")
    p_search.add_argument("query", type=str)
    p_search.add_argument("-k", type=int, default=10)
    p_search.add_argument("--alpha", type=float, default=0.5)
    p_search.set_defaults(func=cmd_search)

    p_eval = subparsers.add_parser("eval", help="Evaluate on test set")
    p_eval.add_argument("--queries", type=str, default=str(DATA_DIR / "queries.tsv"))
    p_eval.add_argument("--qrels", type=str, default=str(DATA_DIR / "qrels.tsv"))
    p_eval.add_argument("--alpha", type=float, default=0.5)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()