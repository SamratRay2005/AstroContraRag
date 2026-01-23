import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import os
import platform
import time
import gc

# --- Configuration ---
INPUT_FILE = "train.csv.gz"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
TITLE_INDEX_FILE = "arxiv_title_index.bin"
ABSTRACT_INDEX_FILE = "arxiv_abstract_index.bin"
METADATA_FILE = "arxiv_metadata.parquet"

# Batch size for a single GPU.
# 128 is generally safe for 24GB VRAM. If you get OOM, reduce to 64 or 32.
BATCH_SIZE = 128 

def main():
    print(f"--- Starting Single-GPU Process on {platform.node()} ---")

    # 1. Force use of GPU 0
    # This ensures PyTorch doesn't accidentally see or try to use other cards
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU detected, falling back to CPU.")
        device = "cpu"

    # 2. Load Data
    print(f"Loading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file {INPUT_FILE} not found.")

    # Optimized load
    df = pd.read_csv(
        INPUT_FILE, 
        compression='gzip', 
        usecols=['arxiv_id', 'title', 'abstract', 'authors'],
        dtype={'arxiv_id': str} 
    )
    df.fillna("", inplace=True)
    print(f"Data loaded: {len(df)} rows.")

    # 3. Prepare Text Lists
    titles = df['title'].tolist()
    abstracts = df['abstract'].tolist()

    # 4. Initialize Model
    # Load directly onto the target device (GPU 0)
    print(f"Loading model: {MODEL_NAME} to {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = 512 

    # 5. Generate Embeddings (Sequential)
    
    # --- Pass A: Encode Titles ---
    print("Encoding TITLE embeddings...")
    start_time = time.time()
    
    title_embeddings = model.encode(
        titles, 
        batch_size=BATCH_SIZE, 
        normalize_embeddings=True, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Titles done in {time.time() - start_time:.2f}s. Shape: {title_embeddings.shape}")
    
    # Clean up input text to free RAM
    del titles
    gc.collect()

    # --- Pass B: Encode Abstracts ---
    print("Encoding ABSTRACT embeddings...")
    start_time = time.time()
    
    abstract_embeddings = model.encode(
        abstracts, 
        batch_size=BATCH_SIZE, 
        normalize_embeddings=True, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Abstracts done in {time.time() - start_time:.2f}s. Shape: {abstract_embeddings.shape}")

    # Clean up input text
    del abstracts
    gc.collect()
    
    # Clear GPU cache (good practice before FAISS if sharing memory)
    if device == "cuda":
        torch.cuda.empty_cache()

    # 6. Build FAISS Indices
    print("Building FAISS Indices...")
    d = title_embeddings.shape[1] 
    
    # Titles
    index_title = faiss.IndexFlatIP(d)
    index_title.add(title_embeddings)
    faiss.write_index(index_title, TITLE_INDEX_FILE)
    print(f"Saved {TITLE_INDEX_FILE}")

    # Abstracts
    index_abstract = faiss.IndexFlatIP(d)
    index_abstract.add(abstract_embeddings)
    faiss.write_index(index_abstract, ABSTRACT_INDEX_FILE)
    print(f"Saved {ABSTRACT_INDEX_FILE}")
    
    # 7. Save Metadata
    print(f"Saving metadata to {METADATA_FILE}...")
    df[['arxiv_id', 'title', 'authors']].to_parquet(METADATA_FILE, engine='pyarrow')

    print("--- Success ---")

if __name__ == "__main__":
    main()