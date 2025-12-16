# app.py - FastAPI Backend for Hybrid Search UI
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from search.bm25 import BM25Index
from search.dense_faiss import DenseFaissIndex
from search.hybrid import hybrid_fusion

app = FastAPI(title="Hybrid Search Engine API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (loaded on startup)
bm25_index: Optional[BM25Index] = None
dense_index: Optional[DenseFaissIndex] = None

MODELS_DIR = Path("models")


class SearchRequest(BaseModel):
    query: str
    k: int = 10
    alpha: float = 0.5


class SearchResult(BaseModel):
    doc_id: str
    score: float
    rank: int
    snippet: str


class SearchResponse(BaseModel):
    query: str
    bm25_results: List[SearchResult]
    dense_results: List[SearchResult]
    hybrid_results: List[SearchResult]
    timings: Dict[str, float]
    disagreements: List[Dict[str, str]]


def load_models():
    """Load pickled models on startup"""
    global bm25_index, dense_index
    
    bm25_path = MODELS_DIR / "bm25.pkl"
    dense_path = MODELS_DIR / "dense.pkl"
    
    if not bm25_path.exists() or not dense_path.exists():
        raise FileNotFoundError(
            "Models not found! Run 'python cli.py build' first."
        )
    
    print("Loading BM25 index...")
    with open(bm25_path, "rb") as f:
        bm25_index = pickle.load(f)
    
    print("Loading Dense index...")
    with open(dense_path, "rb") as f:
        dense_index = pickle.load(f)
    
    print("Models loaded successfully!")


@app.on_event("startup")
async def startup_event():
    load_models()


def get_snippet(text: str, max_chars: int = 200) -> str:
    """Extract first N characters as snippet"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def format_results(
    results: List[tuple],
    index,
    method_name: str
) -> List[SearchResult]:
    """Format search results with snippets"""
    formatted = []
    for rank, (doc_id, score) in enumerate(results, start=1):
        text = index.get_document(doc_id)
        snippet = get_snippet(text)
        formatted.append(
            SearchResult(
                doc_id=doc_id,
                score=round(score, 4),
                rank=rank,
                snippet=snippet
            )
        )
    return formatted


def find_disagreements(
    bm25_results: List[SearchResult],
    dense_results: List[SearchResult],
    hybrid_results: List[SearchResult],
    top_k: int = 5
) -> List[Dict[str, str]]:
    """Find where methods disagree on top results"""
    bm25_top = {r.doc_id for r in bm25_results[:top_k]}
    dense_top = {r.doc_id for r in dense_results[:top_k]}
    hybrid_top = {r.doc_id for r in hybrid_results[:top_k]}
    
    disagreements = []
    
    # BM25 unique docs
    bm25_unique = bm25_top - dense_top
    if bm25_unique:
        disagreements.append({
            "type": "BM25-only",
            "docs": ", ".join(list(bm25_unique)[:3]),
            "description": f"Found by BM25 but not Dense (lexical matches)"
        })
    
    # Dense unique docs
    dense_unique = dense_top - bm25_top
    if dense_unique:
        disagreements.append({
            "type": "Dense-only",
            "docs": ", ".join(list(dense_unique)[:3]),
            "description": f"Found by Dense but not BM25 (semantic matches)"
        })
    
    # Hybrid resolves disagreements
    resolved = hybrid_top & (bm25_unique | dense_unique)
    if resolved:
        disagreements.append({
            "type": "Hybrid-resolved",
            "docs": ", ".join(list(resolved)[:3]),
            "description": f"Hybrid selected from disagreed results"
        })
    
    return disagreements


@app.get("/")
async def root():
    return {
        "message": "Hybrid Search Engine API",
        "status": "running",
        "models": {
            "bm25": bm25_index is not None,
            "dense": dense_index is not None
        }
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform hybrid search and return all results"""
    if bm25_index is None or dense_index is None:
        raise HTTPException(
            status_code=500,
            detail="Models not loaded. Run 'python cli.py build' first."
        )
    
    query = request.query
    k = request.k
    alpha = request.alpha
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Time BM25
    t0 = time.time()
    bm25_results = bm25_index.search(query, k)
    bm25_time = time.time() - t0
    
    # Time Dense
    t0 = time.time()
    dense_results = dense_index.search(query, k)
    dense_time = time.time() - t0
    
    # Time Hybrid
    t0 = time.time()
    hybrid_results = hybrid_fusion(bm25_results, dense_results, alpha=alpha, k=k)
    hybrid_time = time.time() - t0
    
    # Format results
    bm25_formatted = format_results(bm25_results, bm25_index, "BM25")
    dense_formatted = format_results(dense_results, dense_index, "Dense")
    hybrid_formatted = format_results(hybrid_results, bm25_index, "Hybrid")
    
    # Find disagreements
    disagreements = find_disagreements(
        bm25_formatted,
        dense_formatted,
        hybrid_formatted
    )
    
    return SearchResponse(
        query=query,
        bm25_results=bm25_formatted,
        dense_results=dense_formatted,
        hybrid_results=hybrid_formatted,
        timings={
            "bm25_ms": round(bm25_time * 1000, 2),
            "dense_ms": round(dense_time * 1000, 2),
            "hybrid_ms": round(hybrid_time * 1000, 2),
            "total_ms": round((bm25_time + dense_time + hybrid_time) * 1000, 2)
        },
        disagreements=disagreements
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": bm25_index is not None and dense_index is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)