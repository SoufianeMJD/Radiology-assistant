"""
RAG Service - Retrieval Augmented Generation using LlamaIndex
Loads persisted FAISS vector store and provides similarity search
"""
import os
from pathlib import Path
from typing import List, Dict

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import MODEL_CONFIG


class RAGService:
    """RAG Service using LlamaIndex with FAISS vector store"""
    
    def __init__(self):
        self.index = None
        self.embedding_model = None
        self.config = MODEL_CONFIG["rag"]
        self.persist_dir = Path(self.config["persist_dir"])
        
    def load_embedding_model(self):
        """Load HuggingFace embedding model"""
        if self.embedding_model is None:
            print("[RAG] Loading embedding model...")
            model_name = self.config["embedding_model"]
            
            self.embedding_model = HuggingFaceEmbedding(
                model_name=model_name,
                embed_batch_size=self.config["embed_batch_size"]
            )
            
            # Set global LlamaIndex settings
            Settings.embed_model = self.embedding_model
            Settings.chunk_size = self.config["chunk_size"]
            Settings.chunk_overlap = self.config["chunk_overlap"]
            
            print(f"[RAG] Embedding model loaded: {model_name}")
    
    def load_index(self) -> bool:
        """Load existing FAISS index from disk"""
        try:
            if not self.persist_dir.exists():
                print(f"[RAG] Vector store not found at: {self.persist_dir}")
                return False
            
            print("[RAG] Loading persisted vector store...")
            
            # Load embedding model first
            self.load_embedding_model()
            
            # Load the full storage context (not just FAISS)
            # This properly handles the binary FAISS data in JSON files
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
                index = load_index_from_storage(storage_context)
                self.index = index
                
                print(f"[RAG] Vector store loaded successfully")
                print(f"[RAG] Top-k retrieval: {self.config['top_k']}")
                
                return True
                
            except Exception as e:
                print(f"[RAG] Error loading index: {str(e)}")
                print("[RAG] This may be due to binary FAISS data in vector store files")
                print("[RAG] Please ensure you downloaded the latest version from Kaggle")
                
                # Don't try alternative - just fail
                return False
                
        except Exception as e:
            print(f"[RAG] Fatal error loading index: {str(e)}")
            return False
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve most similar reports based on query
        
        Args:
            query: Search query (typically detected pathologies)
            top_k: Number of results to return (overrides config)
            
        Returns:
            List of dicts with report text, metadata, and similarity scores
        """
        # Ensure index is loaded
        if self.index is None:
            if not self.load_index():
                raise RuntimeError(
                    "Vector store not found. Please download vector_database.zip from Kaggle"
                )
        
        try:
            # Perform retrieval using LlamaIndex
            print(f"[RAG] Retrieving similar cases for query: {query[:50]}...")
            
            # Use retriever directly for more control
            retriever = self.index.as_retriever(
                similarity_top_k=top_k or self.config["top_k"]
            )
            
            nodes = retriever.retrieve(query)
            
            # Format results
            results = []
            for rank, node in enumerate(nodes, 1):
                results.append({
                    "rank": rank,
                    "similarity": float(node.score) if node.score else 0.0,
                    "report": node.node.get_content(),
                    "metadata": node.node.metadata
                })
            
            print(f"[RAG] Retrieved {len(results)} similar cases")
            
            return results
            
        except Exception as e:
            print(f"[RAG] Retrieval error: {str(e)}")
            raise


# Singleton instance
rag_service = RAGService()


if __name__ == "__main__":
    # Test the service
    print("Testing RAG Service...")
    
    # Load index
    if rag_service.load_index():
        # Test retrieval
        results = rag_service.retrieve("Cardiomegaly and Edema")
        print(f"\nRetrieved {len(results)} similar cases:")
        for result in results:
            print(f"\nRank {result['rank']} (Similarity: {result['similarity']:.3f})")
            print(result['report'][:200] + "...")
    else:
        print("\n✗ No vector store found!")
        print("Please download vector_database.zip from Kaggle")
