# search/io_utils.py
import json
from typing import Dict, List, Tuple, Set

def load_corpus(path: str) -> Dict[str, str]:
    """
    Expect corpus.jsonl with one JSON per line:
      {"doc_id": "D123", "text": "document text ..."}
    """
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            corpus[obj["doc_id"]] = obj["text"]
    return corpus

def load_queries(path: str) -> Dict[str, str]:
    """
    queries.tsv: qid<TAB>query text
    """
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[qid] = text
    return queries

def load_qrels(path: str) -> Dict[str, Set[str]]:
    """
    qrels.tsv: qid<TAB>doc_id<TAB>rel
    Only keeps rel > 0 as relevant.
    """
    qrels: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, rel_str = parts[0], parts[1], parts[2]
            rel = int(rel_str)
            if rel <= 0:
                continue
            qrels.setdefault(qid, set()).add(doc_id)
    return qrels
