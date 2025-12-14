# search/bm25.py
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Iterable

class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # term -> list[(doc_id, tf)]
        self.index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        # doc_id -> length (in tokens)
        self.doc_len: Dict[str, int] = {}
        self.N: int = 0
        self.avgdl: float = 0.0

        # term -> document frequency
        self.doc_freq: Dict[str, int] = defaultdict(int)

        # store raw text if you want to display snippets
        self.docs: Dict[str, str] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Very simple tokenizer; you can replace with something better later.
        return re.findall(r"\w+", text.lower())

    def add_document(self, doc_id: str, text: str):
        tokens = self._tokenize(text)
        self.docs[doc_id] = text
        self.doc_len[doc_id] = len(tokens)
        self.N += 1

        tf_counts = Counter(tokens)
        for term, tf in tf_counts.items():
            self.index[term].append((doc_id, tf))
            self.doc_freq[term] += 1

    def build_from_corpus(self, corpus: Dict[str, str]):
        for doc_id, text in corpus.items():
            self.add_document(doc_id, text)
        self._finalize()

    def _finalize(self):
        if self.N == 0:
            self.avgdl = 0.0
        else:
            self.avgdl = sum(self.doc_len.values()) / self.N

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        # BM25 idf variant
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        tokens = self._tokenize(query)
        if self.N == 0:
            return []

        scores = defaultdict(float)

        for term in set(tokens):
            if term not in self.index:
                continue
            idf = self._idf(term)
            postings = self.index[term]
            for doc_id, tf in postings:
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score = idf * (tf * (self.k1 + 1)) / denom
                scores[doc_id] += score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def get_document(self, doc_id: str) -> str:
        return self.docs.get(doc_id, "")
