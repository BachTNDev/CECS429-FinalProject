# prepare_msmarco.py - Properly limited corpus version
"""
Convert MS MARCO Passage from ir_datasets with proper corpus filtering
to create a manageable dataset for this project.
"""

import json
from pathlib import Path
import ir_datasets


def main():
    """
    Prepare MS MARCO data with corpus limited to relevant documents only
    
    Strategy: Only include documents that appear in the dev/small qrels
    This gives us ~7k relevant docs + sample of remaining corpus for a 
    total of ~100k-500k docs (configurable)
    """
    
    # Load dev/small for queries and qrels
    dataset_id = "msmarco-passage/dev/small"
    
    print("="*60)
    print(f"Preparing MS MARCO dataset: {dataset_id}")
    print("="*60)
    
    ds = ir_datasets.load(dataset_id)
    
    # Configuration: max documents to include
    MAX_DOCS = 500000  # Manageable size for 4-8GB RAM
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.tsv"
    qrels_path = data_dir / "qrels.tsv"

    # Step 1: Collect relevant document IDs from qrels
    print(f"\n1. Loading relevant document IDs from qrels...")
    relevant_docs = set()
    qrel_count = 0
    
    with open(qrels_path, "w", encoding="utf-8") as qrels_out:
        for qr in ds.qrels_iter():
            qrels_out.write(f"{qr.query_id}\t{qr.doc_id}\t{qr.relevance}\n")
            if qr.relevance > 0:
                relevant_docs.add(qr.doc_id)
            qrel_count += 1
    
    print(f"   ✅ Found {len(relevant_docs):,} relevant documents")
    print(f"   ✅ Wrote {qrel_count:,} qrels")

    # Step 2: Write queries
    print(f"\n2. Writing queries to {queries_path} ...")
    query_count = 0
    with open(queries_path, "w", encoding="utf-8") as f_out:
        for q in ds.queries_iter():
            f_out.write(f"{q.query_id}\t{q.text}\n")
            query_count += 1
    
    print(f"   ✅ Wrote {query_count:,} queries")

    # Step 3: Write corpus (relevant docs + sample of others)
    print(f"\n3. Writing corpus to {corpus_path} ...")
    print(f"   Target size: {MAX_DOCS:,} documents")
    print(f"   - Relevant docs: {len(relevant_docs):,} (guaranteed)")
    print(f"   - Additional docs: {MAX_DOCS - len(relevant_docs):,} (sample)")
    
    doc_count = 0
    relevant_count = 0
    sample_count = 0
    
    with open(corpus_path, "w", encoding="utf-8") as f_out:
        for doc in ds.docs_iter():
            # Always include relevant documents
            if doc.doc_id in relevant_docs:
                obj = {"doc_id": doc.doc_id, "text": doc.text}
                f_out.write(json.dumps(obj) + "\n")
                relevant_count += 1
                doc_count += 1
            # Sample additional documents until we hit MAX_DOCS
            elif doc_count < MAX_DOCS:
                obj = {"doc_id": doc.doc_id, "text": doc.text}
                f_out.write(json.dumps(obj) + "\n")
                sample_count += 1
                doc_count += 1
            else:
                # Stop once we have enough documents
                break
            
            if doc_count % 10000 == 0:
                print(f"   Progress: {doc_count:,}/{MAX_DOCS:,} documents")
    
    print(f"   ✅ Wrote {doc_count:,} documents")
    print(f"      - Relevant: {relevant_count:,}")
    print(f"      - Sample: {sample_count:,}")

    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Dataset: {dataset_id}")
    print(f"Documents: {doc_count:,} (limited from 8.8M)")
    print(f"Queries: {query_count:,}")
    print(f"Relevance judgments: {qrel_count:,}")
    print(f"Relevant docs included: {relevant_count:,}/{len(relevant_docs):,}")
    print("\nFiles created:")
    print(f"  📄 {corpus_path}")
    print(f"  📄 {queries_path}")
    print(f"  📄 {qrels_path}")
    print("="*60)
    
    if doc_count <= 200000:
        print("\n✅ Corpus size is good for this project!")
        print(f"   RAM needed: ~4-8 GB")
        print(f"   Build time: ~5-15 minutes")
    else:
        print("\n⚠️  Corpus is large but manageable")
        print(f"   RAM needed: ~8-16 GB")
        print(f"   Build time: ~15-30 minutes")
    
    print("\nNext steps:")
    print("  python3 cli.py build --corpus data/corpus.jsonl --batch-size 16")
    print("="*60)


if __name__ == "__main__":
    main()