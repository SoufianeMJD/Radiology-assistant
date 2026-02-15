"""
Vector Database Builder Script
One-time script to download dataset and build LlamaIndex vector store

Run this script once to create the persistent vector database.
Subsequent runs will use the cached index.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import kagglehub
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from config import MODEL_CONFIG, DATASET_CONFIG


def download_dataset():
    """Download the Indiana University chest X-ray dataset"""
    print("\n" + "="*60)
    print("DOWNLOADING CHEST X-RAY DATASET")
    print("="*60)
    
    try:
        # Use the exact code from Kaggle website
        print(f"[1/3] Downloading dataset: {DATASET_CONFIG['kaggle_dataset']}")
        path = kagglehub.dataset_download(DATASET_CONFIG['kaggle_dataset'])
        
        dataset_path = Path(path)
        print(f"✓ Dataset downloaded to: {path}")
        
        # List contents
        print("\n[2/3] Dataset contents:")
        for item in dataset_path.rglob("*.csv"):
            print(f"  - {item.name}")
        
        return dataset_path
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {str(e)}")
        raise


def find_reports_file(dataset_path: Path):
    """Find the reports CSV file in the downloaded dataset"""
    print("\n[3/3] Locating reports file...")
    
    # Try common filenames
    possible_names = [
        "indiana_reports.csv",
        "reports.csv",
        "indiana_projections.csv",
        "projections.csv"
    ]
    
    for csv_file in dataset_path.rglob("*.csv"):
        if any(name.lower() in csv_file.name.lower() for name in possible_names):
            print(f"✓ Found reports file: {csv_file.name}")
            return csv_file
    
    # Fallback: find CSV with required columns
    for csv_file in dataset_path.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file, nrows=1)
            if 'findings' in df.columns or 'impression' in df.columns:
                print(f"✓ Found reports file: {csv_file.name}")
                return csv_file
        except:
            continue
    
    raise FileNotFoundError("Could not find reports CSV with 'findings' or 'impression' columns")


def load_reports(csv_path: Path):
    """Load and parse medical reports from CSV"""
    print("\n" + "="*60)
    print("LOADING MEDICAL REPORTS")
    print("="*60)
    
    df = pd.read_csv(csv_path)
    print(f"[1/2] Loaded {len(df)} reports from CSV")
    print(f"[2/2] Columns: {list(df.columns)}")
    
    # Create LlamaIndex Document objects
    documents = []
    
    for idx, row in df.iterrows():
        # Build report text
        report_parts = []
        
        if 'findings' in df.columns and pd.notna(row['findings']):
            report_parts.append(f"Findings: {row['findings']}")
        
        if 'impression' in df.columns and pd.notna(row['impression']):
            report_parts.append(f"Impression: {row['impression']}")
        
        # Skip empty reports
        if not report_parts:
            continue
        
        full_report = "\n".join(report_parts)
        
        # Create Document with metadata
        doc = Document(
            text=full_report,
            metadata={
                "report_id": idx,
                "source": "indiana_university",
                "has_findings": 'findings' in df.columns and pd.notna(row['findings']),
                "has_impression": 'impression' in df.columns and pd.notna(row['impression']),
            }
        )
        documents.append(doc)
    
    print(f"\n✓ Created {len(documents)} Document objects")
    return documents


def build_vector_store(documents: list):
    """Build FAISS vector store with HuggingFace embeddings"""
    print("\n" + "="*60)
    print("BUILDING VECTOR STORE")
    print("="*60)
    
    # Configure LlamaIndex settings
    rag_config = MODEL_CONFIG["rag"]
    
    print(f"[1/5] Initializing HuggingFace embeddings: {rag_config['embedding_model']}")
    embed_model = HuggingFaceEmbedding(
        model_name=rag_config['embedding_model'],
        embed_batch_size=rag_config['embed_batch_size']
    )
    
    # Set global settings
    Settings.embed_model = embed_model
    Settings.chunk_size = rag_config['chunk_size']
    Settings.chunk_overlap = rag_config['chunk_overlap']
    
    print(f"✓ Embedding model loaded")
    print(f"  - Chunk size: {rag_config['chunk_size']} chars")
    print(f"  - Chunk overlap: {rag_config['chunk_overlap']} chars")
    
    # Create FAISS index
    print("\n[2/5] Creating FAISS index...")
    d = 384  # Dimension for all-MiniLM-L6-v2
    faiss_index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
    
    # Create vector store
    print("[3/5] Initializing vector store...")
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Build index from documents
    print(f"[4/5] Indexing {len(documents)} documents...")
    print("  (This may take a few minutes...)")
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    print(f"✓ Index created with {len(documents)} documents")
    
    # Persist to disk
    persist_dir = Path(rag_config['persist_dir'])
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[5/5] Saving index to: {persist_dir}")
    index.storage_context.persist(persist_dir=str(persist_dir))
    
    print(f"✓ Vector store persisted successfully")
    
    return index


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("VECTOR DATABASE BUILDER")
    print("="*60)
    print("\nThis script will:")
    print("  1. Download the Indiana University chest X-ray dataset")
    print("  2. Parse medical reports from the dataset")
    print("  3. Create embeddings using HuggingFace")
    print("  4. Build and persist a FAISS vector store")
    print("\n" + "="*60 + "\n")
    
    try:
        # Check if index already exists
        persist_dir = Path(MODEL_CONFIG["rag"]["persist_dir"])
        if persist_dir.exists() and any(persist_dir.iterdir()):
            print(f"⚠ Vector store already exists at: {persist_dir}")
            response = input("Do you want to rebuild it? (y/N): ")
            if response.lower() != 'y':
                print("\n✓ Using existing vector store")
                return
            print("\n→ Rebuilding vector store...\n")
        
        # Step 1: Download dataset
        dataset_path = download_dataset()
        
        # Step 2: Find reports file
        csv_path = find_reports_file(dataset_path)
        
        # Step 3: Load reports
        documents = load_reports(csv_path)
        
        # Step 4: Build vector store
        index = build_vector_store(documents)
        
        # Success
        print("\n" + "="*60)
        print("✓ VECTOR DATABASE CREATED SUCCESSFULLY")
        print("="*60)
        print(f"\nDocuments indexed: {len(documents)}")
        print(f"Storage location: {MODEL_CONFIG['rag']['persist_dir']}")
        print("\nYou can now start the backend server:")
        print("  python main.py")
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ ERROR")
        print("="*60)
        print(f"\n{str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
