import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Optional
from icecream import ic

class VectorStore:
    def __init__(self, dimension: int = 1536):
        """Initialize the FAISS vector store."""
        self.dimension = dimension
        self.index = None
        self.texts = []
        self.metadatas = []
        self.index_file = "vector_index.faiss"
        self.metadata_file = "metadata.pkl"
        self._load_index()
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        """Add documents with their embeddings to the vector store."""
        try:
            if not texts or not embeddings:
                return
            embeddings_array = np.array(embeddings, dtype=np.float32)
            if self.index is None:
                self.dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                self.texts = []
                self.metadatas = []
            ic("Adding documents to vector store")
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            ic("Documents added to index")
            self.texts.extend(texts)
            self.metadatas.extend(metadatas)
            ic("Texts and metadata updated in vector store")
            self._save_index()
            ic("Index saved successfully")
        except Exception as e:
            raise Exception(f"Error adding documents to vector store: {str(e)}")
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using the query embedding."""
        try:
            if self.index is None or len(self.texts) == 0:
                return []
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            k = min(k, len(self.texts))
            scores, indices = self.index.search(query_array, k)
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.texts):
                    results.append({
                        'text': self.texts[idx],
                        'metadata': self.metadatas[idx],
                        'score': float(score),
                        'rank': i + 1
                    })
            return results
        except Exception as e:
            raise Exception(f"Error during similarity search: {str(e)}")
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self.index = None
        self.texts = []
        self.metadatas = []
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_documents': len(self.texts),
            'vector_dimensions': self.dimension if self.index else 0,
            'index_size': self.index.ntotal if self.index else 0
        }
    
    def _save_index(self) -> None:
        """Save the FAISS index and metadata to disk."""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump({
                    'texts': self.texts,
                    'metadatas': self.metadatas,
                    'dimension': self.dimension
                }, f)
        except Exception as e:
            print(f"Error saving index: {str(e)}")
    
    def _load_index(self) -> None:
        """Load the FAISS index and metadata from disk."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.texts = data.get('texts', [])
                    self.metadatas = data.get('metadatas', [])
                    self.dimension = data.get('dimension', 1536)
            if os.path.exists(self.index_file) and self.texts:
                self.index = faiss.read_index(self.index_file)
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            self.index = None
            self.texts = []
            self.metadatas = []
