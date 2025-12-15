# Hybrid Search Engine from Scratch

A production-grade hybrid search engine combining BM25 (lexical) and Dense (semantic) retrieval with FAISS, evaluated on MS MARCO.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Query Input                       │
└────────────────┬────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   ┌─────────┐      ┌──────────┐
   │  BM25   │      │  Dense   │
   │ Index   │      │  FAISS   │
   └────┬────┘      └────┬─────┘
        │                │
        └────────┬────────┘
                 ▼
          ┌──────────────┐
          │ Hybrid Fusion│
          │ (α-weighted) │
          └──────┬───────┘
                 ▼
            Final Results
```

### Components

- **BM25 Index**: Custom inverted index with TF-IDF scoring
- **Dense Index**: Sentence-BERT embeddings + FAISS HNSW
- **Hybrid Fusion**: Min-max normalized linear combination
- **Evaluation**: nDCG@10, MRR@10, Recall@100 on MS MARCO

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Download MS MARCO passage dev/small (~500k docs)
python prepare_msmarco.py
```

This creates:
- `data/corpus.jsonl` - Document collection
- `data/queries.tsv` - Test queries
- `data/qrels.tsv` - Relevance judgments

### 3. Build Indexes

```bash
# Build BM25 and Dense FAISS indexes
python cli.py build --corpus data/corpus.jsonl
```

**Time**: ~5-15 minutes (CPU), ~2-5 minutes (GPU)
**Output**: `models/bm25.pkl`, `models/dense.pkl`

### 4. Run Evaluation

```bash
# Evaluate all three methods
python cli.py eval \
  --queries data/queries.tsv \
  --qrels data/qrels.tsv \
  --alpha 0.5
```

**Expected Results** (MS MARCO dev/small):
| Method  | nDCG@10 | MRR@10 | Recall@100 |
|---------|---------|--------|------------|
| BM25    | 0.25-0.28 | 0.18-0.20 | 0.65-0.70 |
| Dense   | 0.28-0.32 | 0.20-0.23 | 0.70-0.75 |
| Hybrid  | 0.30-0.34 | 0.22-0.25 | 0.75-0.80 |

### 5. Interactive Search

```bash
# Search with different methods
python cli.py search "what causes rain" -k 10 --alpha 0.5
```

## 📊 Experiments & Ablations

### 1. Tune Hybrid Alpha

```bash
# Test different BM25/Dense weights
for alpha in 0.0 0.25 0.5 0.75 1.0; do
  echo "Alpha: $alpha"
  python cli.py eval --alpha $alpha
done
```

### 2. BM25 Parameters

Edit `search/bm25.py`:
```python
# Default: k1=1.2, b=0.75
bm25 = BM25Index(k1=1.5, b=0.8)  # Try different values
```

### 3. FAISS Parameters

Edit `search/dense_faiss.py`:
```python
# HNSW parameters
hnsw_m = 32          # Connections per layer (16, 32, 64)
efSearch = 64        # Search beam width (32, 64, 128)
efConstruction = 200 # Build beam width (100, 200, 400)
```

## 🔍 Project Structure

```
.
├── search/
│   ├── bm25.py           # BM25 inverted index
│   ├── dense_faiss.py    # Dense retrieval + FAISS
│   ├── hybrid.py         # Score fusion
│   ├── eval.py           # Evaluation metrics
│   └── io_utils.py       # Data loading utilities
├── cli.py                # Command-line interface
├── prepare_msmarco.py    # Data preparation
├── requirements.txt      # Dependencies
├── data/                 # Generated data files
└── models/               # Saved indexes
```

## 🧪 Implementation Details

### BM25 Index
- **Algorithm**: Okapi BM25 with standard parameters (k1=1.2, b=0.75)
- **Tokenization**: Simple regex-based `\w+` tokenizer
- **Storage**: In-memory inverted index with postings lists

### Dense Index
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Index**: FAISS HNSW (Hierarchical Navigable Small World)
- **Similarity**: Cosine similarity (normalized L2)
- **Config**: M=32, efSearch=64, efConstruction=200

### Hybrid Fusion
- **Method**: Linear combination after min-max normalization
- **Formula**: `score = α·BM25_norm + (1-α)·Dense_norm`
- **Default**: α=0.5 (equal weighting)

## 📈 Performance Benchmarks

**Hardware**: Apple M1 / Intel i7 / AWS t3.medium

| Operation | Time (CPU) | Time (GPU) |
|-----------|------------|------------|
| Build BM25 | ~2 min | N/A |
| Build Dense | ~10 min | ~3 min |
| Search BM25 | ~5 ms | N/A |
| Search Dense | ~15 ms | ~8 ms |
| Hybrid Fusion | ~20 ms | ~10 ms |

## 🎯 Future Improvements

### Easy Wins
- [ ] Better tokenization (BERT tokenizer, stemming)
- [ ] Query expansion (RM3, pseudo-relevance feedback)
- [ ] More fusion methods (RRF, learned fusion)

### Medium Effort
- [ ] Cross-encoder reranker (monoT5-small on top-100)
- [ ] ColBERT late interaction
- [ ] Quantization (FAISS IVF-PQ) for 10x memory reduction

### Advanced
- [ ] Learned sparse retrieval (SPLADE)
- [ ] Multi-vector representations
- [ ] Neural fusion model

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'faiss'"
```bash
# Use CPU version
pip install faiss-cpu

# Or GPU version (requires CUDA)
pip install faiss-gpu
```

### "Out of memory" during indexing
Reduce batch size in `dense_faiss.py`:
```python
dense.build_from_corpus(corpus, batch_size=32)  # Default: 64
```

### Slow FAISS search
Increase HNSW efSearch for better quality:
```python
self.index.hnsw.efSearch = 128  # Default: 64
```

## 📚 References

- **BM25**: Robertson & Zaragoza (2009) - "The Probabilistic Relevance Framework: BM25 and Beyond"
- **FAISS**: Johnson et al. (2017) - "Billion-scale similarity search with GPUs"
- **Sentence-BERT**: Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- **MS MARCO**: Nguyen et al. (2016) - "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset"

## 📝 License

MIT License - feel free to use for your projects!

## 🙋 Contributing

Pull requests welcome! Areas of interest:
- Better tokenization/preprocessing
- Additional fusion strategies
- Performance optimizations
- UI improvements