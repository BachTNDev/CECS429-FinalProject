# prepare_msmarco.py - Fixed version with proper dataset selection
"""
Convert MS MARCO Passage (dev/small) from ir_datasets
into our engine's formats:
  - data/corpus.jsonl
  - data/queries.tsv
  - data/qrels.tsv
"""

import json
from pathlib import Path
import ir_datasets


def main():
    """
    Prepare MS MARCO data for the search engine
    
    Dataset options (in order of size):
    - msmarco-passage/dev/small  (~500k docs, 6,900 queries) - RECOMMENDED
    - msmarco-passage/train      (~8.8M docs, 500k queries) - TOO LARGE
    - msmarco-passage/dev        (~8.8M docs, 6,900 queries) - TOO LARGE
    """
    
    # USE THE SMALL VERSION!
    dataset_id = "msmarco-passage/dev/small"
    
    print("="*60)
    print(f"Preparing MS MARCO dataset: {dataset_id}")
    print("="*60)
    
    ds = ir_datasets.load(dataset_id)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.tsv"
    qrels_path = data_dir / "qrels.tsv"

    # 1) Write corpus.jsonl
    print(f"\n1. Writing corpus to {corpus_path} ...")
    doc_count = 0
    with open(corpus_path, "w", encoding="utf-8") as f_out:
        for doc in ds.docs_iter():
            obj = {
                "doc_id": doc.doc_id,
                "text": doc.text,
            }
            f_out.write(json.dumps(obj) + "\n")
            doc_count += 1
            
            if doc_count % 50000 == 0:
                print(f"   Progress: {doc_count} documents")
    
    print(f"   ✅ Done: {doc_count} documents")

    # 2) Write queries.tsv
    print(f"\n2. Writing queries to {queries_path} ...")
    query_count = 0
    with open(queries_path, "w", encoding="utf-8") as f_out:
        for q in ds.queries_iter():
            f_out.write(f"{q.query_id}\t{q.text}\n")
            query_count += 1
    
    print(f"   ✅ Done: {query_count} queries")

    # 3) Write qrels.tsv
    print(f"\n3. Writing qrels to {qrels_path} ...")
    qrel_count = 0
    with open(qrels_path, "w", encoding="utf-8") as f_out:
        for qr in ds.qrels_iter():
            f_out.write(f"{qr.query_id}\t{qr.doc_id}\t{qr.relevance}\n")
            qrel_count += 1
    
    print(f"   ✅ Done: {qrel_count} relevance judgments")

    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Dataset: {dataset_id}")
    print(f"Documents: {doc_count:,}")
    print(f"Queries: {query_count:,}")
    print(f"Relevance judgments: {qrel_count:,}")
    print("\nFiles created:")
    print(f"  📄 {corpus_path}")
    print(f"  📄 {queries_path}")
    print(f"  📄 {qrels_path}")
    print("="*60)
    
    # Validate document count
    if doc_count > 1_000_000:
        print("\n⚠️  WARNING: Document count seems too high!")
        print(f"   Expected ~500k documents, got {doc_count:,}")
        print("   This will require a lot of RAM to build indexes.")
        print("\n💡 TIP: Use msmarco-passage/dev/small instead")
    else:
        print("\n✅ Document count looks good for this project!")
    
    print("\nNext steps:")
    print("  python3 cli.py build --corpus data/corpus.jsonl")
    print("="*60)


if __name__ == "__main__":
    main()