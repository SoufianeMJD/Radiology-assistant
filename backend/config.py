"""
Configuration management for the Radiological AI Assistant
"""
import os
import torch
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
FAISS_INDEX_DIR.mkdir(exist_ok=True)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[CONFIG] Using device: {DEVICE}")

# Model configurations
MODEL_CONFIG = {
    "vision": {
        "model_name": "densenet121-res224-all",
        "confidence_threshold": 0.5,
    },
    "rag": {
        # LlamaIndex configuration
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # Local HuggingFace embeddings
        "embed_batch_size": 10,
        "top_k": 3,
        "chunk_size": 512,  # Characters per document chunk
        "chunk_overlap": 50,  # Overlap between chunks
        
        # Storage paths
        "vector_store_dir": str(FAISS_INDEX_DIR),
        "persist_dir": str(FAISS_INDEX_DIR / "storage"),
        
        # Legacy paths (deprecated, for migration reference)
        "index_path": str(FAISS_INDEX_DIR / "medical_reports.index"),
        "metadata_path": str(FAISS_INDEX_DIR / "metadata.pkl"),
    },
    "llm": {
        "provider": "gemini",
        "model_name": "gemini-flash-latest",  # Fixed: added models/ prefix
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "api_timeout": 30,
    }
}

# API configurations
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": {".jpg", ".jpeg", ".png"},
}

# Dataset configuration
DATASET_CONFIG = {
    "kaggle_dataset": "raddar/chest-xrays-indiana-university",
    "reports_filename": "indiana_reports.csv",
}
