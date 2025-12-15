# search/dense_faiss.py - GPU-optimized version
from typing import Dict, List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

class DenseFaissIndex:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hnsw_m: int = 32,
        use_gpu: bool = True
    ):
        """
        Initialize Dense FAISS index with optional GPU support
        
        Args:
            model_name: Sentence transformer model to use
            hnsw_m: HNSW parameter (connections per layer)
            use_gpu: Try to use GPU if available (default: True)
        """
        self.model_name = model_name
        self.hnsw_m = hnsw_m
        
        # Check GPU availability
        self.gpu_available = torch.cuda.is_available() and use_gpu
        
        if self.gpu_available:
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Using CPU (no GPU detected or use_gpu=False)")
        
        # Initialize sentence transformer
        print(f"📦 Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        
        # Move model to GPU if available
        if self.gpu_available:
            self.model = self.model.to('cuda')
            print(f"   ✅ Model on GPU")
        else:
            print(f"   ✅ Model on CPU")
        
        # Initialize index (will be set up in build_from_corpus)
        self.index = None
        self.faiss_gpu_available = False
        self.gpu_resources = None
        
        # Storage
        self.doc_ids: List[str] = []
        self.docs: Dict[str, str] = {}

    def _setup_index(self):
        """Set up FAISS index with GPU support if available"""
        # Create CPU index first
        cpu_index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m)
        cpu_index.hnsw.efSearch = 64
        cpu_index.hnsw.efConstruction = 200
        
        # Try to use GPU for FAISS
        if self.gpu_available:
            try:
                # Check if FAISS GPU is available
                if hasattr(faiss, 'StandardGpuResources'):
                    self.gpu_resources = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(
                        self.gpu_resources, 
                        0,  # GPU device 0
                        cpu_index
                    )
                    self.faiss_gpu_available = True
                    print(f"   ✅ FAISS index on GPU")
                    return
                else:
                    print(f"   ⚠️  FAISS-GPU not installed, using CPU for index")
            except Exception as e:
                print(f"   ⚠️  Failed to move FAISS to GPU: {e}")
                print(f"      Using CPU for index")
        
        # Fall back to CPU
        self.index = cpu_index
        self.faiss_gpu_available = False
        print(f"   ✅ FAISS index on CPU")

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings (uses GPU if available)
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of embeddings (normalized)
        """
        batch_size = 128 if self.gpu_available else 32
        
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype("float32")

    def build_from_corpus(self, corpus: Dict[str, str], batch_size: int = None):
        """
        Build index from corpus with GPU acceleration
        
        Args:
            corpus: Dict mapping doc_id to text
            batch_size: Encoding batch size (auto-selected based on device)
        """
        if batch_size is None:
            # Use larger batches on GPU
            batch_size = 128 if self.gpu_available else 32
        
        print(f"\n{'='*60}")
        print(f"Building Dense Index: {len(corpus)} documents")
        print(f"{'='*60}")
        print(f"Encoding batch size: {batch_size}")
        
        # Set up FAISS index
        self._setup_index()
        
        doc_ids = list(corpus.keys())
        texts = [corpus[doc_id] for doc_id in doc_ids]

        self.doc_ids = []
        self.docs = corpus

        print(f"Encoding documents...")
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_ids = doc_ids[start:end]
            
            # Encode on GPU/CPU
            emb = self._encode(batch_texts)
            
            # Add to index
            self.index.add(emb)
            self.doc_ids.extend(batch_ids)
            
            # Progress update
            if (start // batch_size) % 10 == 0 or end == len(texts):
                print(f"  Progress: {end}/{len(texts)} documents "
                      f"({end/len(texts)*100:.1f}%)")
        
        print(f"{'='*60}")
        print(f"✅ Index built successfully")
        print(f"   Documents: {len(self.doc_ids)}")
        print(f"   Embedding model: {'GPU' if self.gpu_available else 'CPU'}")
        print(f"   FAISS index: {'GPU' if self.faiss_gpu_available else 'CPU'}")
        print(f"{'='*60}\n")

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_from_corpus first.")
        
        # Encode query (on GPU if available)
        q_emb = self._encode([query])
        
        # Search (on GPU if available)
        D, I = self.index.search(q_emb, k)
        
        # Format results
        scores_ids: List[Tuple[str, float]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            doc_id = self.doc_ids[idx]
            scores_ids.append((doc_id, float(score)))

        return scores_ids

    def get_document(self, doc_id: str) -> str:
        """Get document text by ID"""
        return self.docs.get(doc_id, "")
    
    def get_config_info(self) -> dict:
        """Get information about current configuration"""
        return {
            "model": self.model_name,
            "embedding_dim": self.dim,
            "hnsw_m": self.hnsw_m,
            "device": "GPU" if self.gpu_available else "CPU",
            "faiss_device": "GPU" if self.faiss_gpu_available else "CPU",
            "gpu_name": torch.cuda.get_device_name(0) if self.gpu_available else None,
            "num_docs": len(self.doc_ids)
        }
    
    def print_config(self):
        """Print current configuration"""
        config = self.get_config_info()
        print("\nDense Index Configuration:")
        print("-" * 40)
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("-" * 40)