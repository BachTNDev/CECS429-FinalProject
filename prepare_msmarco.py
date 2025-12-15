# prepare_msmarco.py - FIXED: Ensures all relevant docs are included
"""
Convert MS MARCO Passage from ir_datasets with TWO-PASS approach:
1. First pass: collect all relevant document IDs from qrels
2. Second pass: write corpus with ALL relevant docs + sample of others
"""

import json
from pathlib import Path
import ir_datasets


def main():
    """
    Prepare MS MARCO data ensuring ALL relevant documents are included
    
    Strategy:
    1. Load qrels and collect all relevant doc IDs
    2. Load full corpus and write:
       - ALL docs that appear in qrels (guaranteed)
       - Sample additional docs to reach target size
    """
    
    dataset_id = "msmarco-passage/dev/small"
    
    print("="*70)
    print(f"Preparing MS MARCO dataset: {dataset_id}")
    print("="*70)
    
    ds = ir_datasets.load(dataset_id)
    
    # Configuration
    TARGET_CORPUS_SIZE = 100000  # Smaller = faster, still good results
    # For better results, use 500000, but needs more RAM/time
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.tsv"
    qrels_path = data_dir / "qrels.tsv"

    # ========================================================================
    # PASS 1: Collect relevant document IDs from qrels
    # ========================================================================
    print(f"\n📋 PASS 1: Collecting relevant document IDs...")
    relevant_docs = set()
    qrel_count = 0
    
    with open(qrels_path, "w", encoding="utf-8") as qrels_out:
        for qr in ds.qrels_iter():
            qrels_out.write(f"{qr.query_id}\t{qr.doc_id}\t{qr.relevance}\n")
            if qr.relevance > 0:
                relevant_docs.add(qr.doc_id)
            qrel_count += 1
    
    print(f"   ✅ Found {len(relevant_docs):,} unique relevant documents")
    print(f"   ✅ Wrote {qrel_count:,} qrels entries")
    
    # ========================================================================
    # Write queries
    # ========================================================================
    print(f"\n📝 Writing queries...")
    query_count = 0
    with open(queries_path, "w", encoding="utf-8") as f_out:
        for q in ds.queries_iter():
            f_out.write(f"{q.query_id}\t{q.text}\n")
            query_count += 1
    
    print(f"   ✅ Wrote {query_count:,} queries")

    # ========================================================================
    # PASS 2: Write corpus ensuring ALL relevant docs are included
    # ========================================================================
    print(f"\n📚 PASS 2: Writing corpus...")
    print(f"   Target: {TARGET_CORPUS_SIZE:,} documents")
    print(f"   Strategy:")
    print(f"     1. Include ALL {len(relevant_docs):,} relevant docs (guaranteed)")
    print(f"     2. Sample {TARGET_CORPUS_SIZE - len(relevant_docs):,} additional docs")
    print()
    
    relevant_written = set()
    sample_written = 0
    doc_count = 0
    
    # Track which relevant docs we've seen
    relevant_found = set()
    
    with open(corpus_path, "w", encoding="utf-8") as f_out:
        for i, doc in enumerate(ds.docs_iter()):
            # Progress indicator for the long iteration
            if (i + 1) % 100000 == 0:
                print(f"   Scanning: {i+1:,} docs | "
                      f"Written: {doc_count:,}/{TARGET_CORPUS_SIZE:,} | "
                      f"Relevant: {len(relevant_written):,}/{len(relevant_docs):,}")
            
            # ALWAYS include relevant documents
            if doc.doc_id in relevant_docs:
                if doc.doc_id not in relevant_written:
                    obj = {"doc_id": doc.doc_id, "text": doc.text}
                    f_out.write(json.dumps(obj) + "\n")
                    relevant_written.add(doc.doc_id)
                    relevant_found.add(doc.doc_id)
                    doc_count += 1
            
            # Sample additional documents if we haven't hit target
            elif doc_count < TARGET_CORPUS_SIZE:
                obj = {"doc_id": doc.doc_id, "text": doc.text}
                f_out.write(json.dumps(obj) + "\n")
                sample_written += 1
                doc_count += 1
            
            # Stop if we have all relevant docs AND hit target size
            if len(relevant_written) == len(relevant_docs) and doc_count >= TARGET_CORPUS_SIZE:
                print(f"\n   ✅ Target reached! Stopping iteration.")
                break
    
    print(f"\n   ✅ Corpus written: {doc_count:,} documents")
    print(f"      - Relevant docs: {len(relevant_written):,}/{len(relevant_docs):,}")
    print(f"      - Sample docs: {sample_written:,}")
    
    # ========================================================================
    # Verification & Summary
    # ========================================================================
    missing_relevant = relevant_docs - relevant_written
    
    print("\n" + "="*70)
    print("VERIFICATION:")
    print("="*70)
    
    if missing_relevant:
        print(f"⚠️  WARNING: {len(missing_relevant)} relevant docs NOT in corpus!")
        print(f"   This will hurt evaluation metrics!")
        print(f"   Missing docs: {list(missing_relevant)[:5]}...")
        print(f"\n   SOLUTION: The corpus needs to be larger or iterate more.")
    else:
        print(f"✅ SUCCESS: ALL {len(relevant_docs):,} relevant docs are in corpus!")
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"Dataset: {dataset_id}")
    print(f"Total corpus size: {doc_count:,}")
    print(f"  - Relevant docs: {len(relevant_written):,}")
    print(f"  - Sample docs: {sample_written:,}")
    print(f"Queries: {query_count:,}")
    print(f"Qrels: {qrel_count:,}")
    print(f"\nRelevant doc coverage: {len(relevant_written)}/{len(relevant_docs)} "
          f"({len(relevant_written)/len(relevant_docs)*100:.1f}%)")
    
    print("\nFiles created:")
    print(f"  📄 {corpus_path}")
    print(f"  📄 {queries_path}")
    print(f"  📄 {qrels_path}")
    
    # Estimate performance
    coverage = len(relevant_written) / len(relevant_docs) if relevant_docs else 0
    
    print("\n" + "="*70)
    print("EXPECTED PERFORMANCE:")
    print("="*70)
    
    if coverage >= 0.95:
        print("✅ EXCELLENT corpus coverage!")
        print("   Expected metrics:")
        print("     - BM25:   nDCG@10 ~0.20-0.25")
        print("     - Dense:  nDCG@10 ~0.25-0.30")
        print("     - Hybrid: nDCG@10 ~0.28-0.32")
    elif coverage >= 0.80:
        print("⚠️  GOOD corpus coverage")
        print("   Expected metrics:")
        print("     - BM25:   nDCG@10 ~0.15-0.20")
        print("     - Dense:  nDCG@10 ~0.18-0.23")
        print("     - Hybrid: nDCG@10 ~0.20-0.25")
    else:
        print("❌ POOR corpus coverage - metrics will be low!")
        print("   Consider increasing TARGET_CORPUS_SIZE")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Rebuild indexes:")
    print("   python cli.py build --corpus data/corpus.jsonl --batch-size 32")
    print("\n2. Run evaluation:")
    print("   python cli.py eval")
    print("\n3. For comprehensive analysis:")
    print("   python analysis.py")
    print("="*70)


if __name__ == "__main__":
    main()