# prepare_msmarco.py
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
    # 1) Load the dataset subset
    # Options you can try:
    #   "msmarco-passage/dev/small"  (nice small benchmark)
    #   "msmarco-passage/dev"        (larger)
    dataset_id = "msmarco-passage/dev/small"
    ds = ir_datasets.load(dataset_id)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.tsv"
    qrels_path = data_dir / "qrels.tsv"

    # 2) Write corpus.jsonl
    #    docs_iter() yields objects with .doc_id and .text
    print(f"Writing corpus to {corpus_path} ...")
    with open(corpus_path, "w", encoding="utf-8") as f_out:
        for doc in ds.docs_iter():
            obj = {
                "doc_id": doc.doc_id,
                "text": doc.text,
            }
            f_out.write(json.dumps(obj) + "\n")

    # 3) Write queries.tsv
    #    queries_iter() yields objects with .query_id and .text
    print(f"Writing queries to {queries_path} ...")
    with open(queries_path, "w", encoding="utf-8") as f_out:
        for q in ds.queries_iter():
            f_out.write(f"{q.query_id}\t{q.text}\n")

    # 4) Write qrels.tsv
    #    qrels_iter() yields objects with .query_id, .doc_id, .relevance
    print(f"Writing qrels to {qrels_path} ...")
    with open(qrels_path, "w", encoding="utf-8") as f_out:
        for qr in ds.qrels_iter():
            # Keep all relevance > 0 (standard)
            f_out.write(f"{qr.query_id}\t{qr.doc_id}\t{qr.relevance}\n")

    print("Done. Files generated in ./data/")
    print(f"  - {corpus_path}")
    print(f"  - {queries_path}")
    print(f"  - {qrels_path}")


if __name__ == "__main__":
    main()
