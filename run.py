#!/usr/bin/env python3
"""
Retrieval + Contradiction Reranker (CSV Hotfix Version)

Key behavior:
- Use a SentenceTransformer (E) for retrieval + FAISS (title/abstract indices).
- Use SparseCL checkpoint (Es) for Hoyer sparsity.
- Rerank top-K candidates by: final_score = cos_E(query, doc) + ALPHA * Hoyer_Es(query, doc)
- FIX: Loads 'train.csv.gz' directly to access abstracts (bypassing broken parquet).
- FIX: Deduplicates candidates based on title.
"""

import os
import sys
import math
import faiss
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------
# Config
# -----------------------
TITLE_INDEX_PATH = "arxiv_title_index.bin"
ABSTRACT_INDEX_PATH = "arxiv_abstract_index.bin"
SPARSE_CHECKPOINT = "sparsecl_checkpoint.pth"

# CHANGED: Point to the original CSV instead of the broken parquet
DATA_FILE_PATH = "train.csv.gz" 

# optional precomputed embeddings (recommended)
E_CORPUS_PATH = "E_corpus.npy"    
ES_CORPUS_PATH = "Es_corpus.npy" 

RETRIEVAL_MODEL_NAME = "BAAI/bge-base-en-v1.5"   
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1" 

TOP_K_CANDIDATES = 500
ALPHA = 5.88454
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
TITLE_THRESHOLD = 0.57
ABSTRACT_THRESHOLD = 0.71
EPS = 1e-8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = True if DEVICE.type == "cuda" else False

# -----------------------
# Utilities & Models
# -----------------------
class SparseSentenceEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(cls, p=2, dim=1)

def calculate_hoyer_vector(emb_a: torch.Tensor, emb_b: torch.Tensor):
    diff = emb_a - emb_b
    d = diff.shape[1]
    l1 = torch.norm(diff, p=1, dim=1)
    l2 = torch.norm(diff, p=2, dim=1) + EPS
    sqrt_d = math.sqrt(d)
    hoyer = (sqrt_d - (l1 / l2)) / (sqrt_d - 1.0)
    return torch.clamp(hoyer, 0.0, 1.0)

# -----------------------
# LLM wrapper
# -----------------------
class MistralResponder:
    def __init__(self):
        self.model = None
        try:
            print("Loading Mistral tokenizer+model (this may be heavy)...")
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                dtype=torch.float16 if USE_FP16 else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        except Exception as e:
            print(f"Warning: Could not load LLM ({e}). Skipping LLM features.")
            self.model = None
            self.tokenizer = None

    def generate_response(self, user_query, best_context, contra_context, max_new_tokens=256):
        if self.model is None:
            return "LLM not available."
        prompt = f"""[INST] You are an expert assistant.
User Query: "{user_query}"

Best Match (consensus):
{best_context}

Contradictory / Alternative:
{contra_context}

Provide a clear answer that:
- Summarizes consensus
- Explains the contradiction and why it might exist
- Gives final takeaway
[/INST]"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in text:
            return text.split("[/INST]")[-1].strip()
        return text

# -----------------------
# Loading resources
# -----------------------
def load_resources():
    print("Loading FAISS indices...")
    if not os.path.exists(TITLE_INDEX_PATH) or not os.path.exists(ABSTRACT_INDEX_PATH):
        raise FileNotFoundError("Missing FAISS index files.")
    idx_title = faiss.read_index(TITLE_INDEX_PATH)
    idx_abstract = faiss.read_index(ABSTRACT_INDEX_PATH)

    # ---------------------------------------------------------
    # UPDATED: Load CSV directly (Slow but complete)
    # ---------------------------------------------------------
    print(f"Loading metadata from {DATA_FILE_PATH} (this might take a moment)...")
    if not os.path.exists(DATA_FILE_PATH):
         raise FileNotFoundError(f"Missing data file: {DATA_FILE_PATH}")
    
    # We must replicate the load logic from encode.py to ensure indices match
    df = pd.read_csv(
        DATA_FILE_PATH, 
        compression='gzip', 
        usecols=['arxiv_id', 'title', 'abstract', 'authors'],
        dtype={'arxiv_id': str} 
    )
    df.fillna("", inplace=True)
    # ---------------------------------------------------------

    print("Loading retrieval model (SentenceTransformer)...")
    retrieval_model = SentenceTransformer(RETRIEVAL_MODEL_NAME)

    print("Loading sparse model (SparseCL checkpoint)...")
    sparse_model = SparseSentenceEncoder(RETRIEVAL_MODEL_NAME).to(DEVICE)
    if os.path.exists(SPARSE_CHECKPOINT):
        try:
            ckpt = torch.load(SPARSE_CHECKPOINT, map_location=DEVICE, weights_only=True)
        except:
            ckpt = torch.load(SPARSE_CHECKPOINT, map_location=DEVICE, weights_only=False)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        sparse_model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError("SparseCL checkpoint not found.")
    sparse_model.eval()

    sparse_tokenizer = AutoTokenizer.from_pretrained(RETRIEVAL_MODEL_NAME)

    E_corpus = None
    Es_corpus = None
    if os.path.exists(E_CORPUS_PATH):
        print(f"Loading precomputed dense embeddings: {E_CORPUS_PATH}")
        E_corpus = np.load(E_CORPUS_PATH, mmap_mode="r")
    if os.path.exists(ES_CORPUS_PATH):
        print(f"Loading precomputed sparse embeddings: {ES_CORPUS_PATH}")
        Es_corpus = np.load(ES_CORPUS_PATH, mmap_mode="r")

    llm = None
    try:
        llm = MistralResponder()
    except Exception:
        llm = None

    return idx_title, idx_abstract, df, retrieval_model, sparse_model, sparse_tokenizer, E_corpus, Es_corpus, llm

# -----------------------
# Helpers
# -----------------------
def query_index(index, query_vector_np: np.ndarray, k: int, threshold: float):
    D, I = index.search(query_vector_np.astype(np.float32), k)
    matches = {}
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        if score >= threshold:
            matches[int(idx)] = float(score)
    return matches

@torch.inference_mode()
def compute_sparse_embeddings_for_texts(texts, tokenizer, sparse_model, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        inp = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt", max_length=256).to(DEVICE)
        emb = sparse_model(inp['input_ids'], inp['attention_mask'])
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)

# -----------------------
# Rerank
# -----------------------
def rerank_topk_and_find_contradiction(
    query_text, best_idx, best_source,
    candidate_indices, candidate_texts,
    retrieval_model, sparse_model, sparse_tokenizer,
    E_corpus=None, Es_corpus=None, alpha=ALPHA, device=DEVICE
):
    k = len(candidate_indices)
    if k == 0:
        return None, None, {}

    # 1) Dense cosine
    full_query = QUERY_INSTRUCTION + query_text
    E_query = retrieval_model.encode([full_query], normalize_embeddings=True)

    if E_corpus is not None:
        E_topk = np.asarray(E_corpus[candidate_indices])
        cos_scores = (E_topk @ E_query.T).squeeze(1)
    else:
        dense_chunks = []
        batch = 64
        for i in range(0, k, batch):
            batch_texts = candidate_texts[i:i+batch]
            enc = retrieval_model.encode(batch_texts, normalize_embeddings=True)
            dense_chunks.append(enc)
        Ek = np.vstack(dense_chunks)
        cos_scores = (Ek @ E_query.T).squeeze(1)

    # 2) Hoyer sparsity
    if Es_corpus is not None:
        Es_topk = np.asarray(Es_corpus[candidate_indices])
        inputs = sparse_tokenizer([full_query], padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
        with torch.no_grad():
            Es_query = sparse_model(inputs['input_ids'], inputs['attention_mask']).cpu().numpy()
        Es_topk_t = torch.from_numpy(Es_topk).to(device).float()
        Es_query_t = torch.from_numpy(Es_query).to(device).float()
        hoyer_scores = calculate_hoyer_vector(Es_topk_t, Es_query_t.expand_as(Es_topk_t)).cpu().numpy()
    else:
        Es_topk = compute_sparse_embeddings_for_texts(candidate_texts, sparse_tokenizer, sparse_model, batch_size=64)
        inputs = sparse_tokenizer([full_query], padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
        with torch.no_grad():
            Es_query = sparse_model(inputs['input_ids'], inputs['attention_mask']).cpu().numpy()
        Es_topk_t = torch.from_numpy(Es_topk).to(device).float()
        Es_query_t = torch.from_numpy(Es_query).to(device).float()
        hoyer_scores = calculate_hoyer_vector(Es_topk_t, Es_query_t.expand_as(Es_topk_t)).cpu().numpy()

    # 3) Combine
    final_scores = cos_scores + alpha * hoyer_scores

    best_local = int(np.argmax(final_scores))
    best_global_idx = candidate_indices[best_local]
    best_score = float(final_scores[best_local])

    details = {
        "cos_scores": cos_scores,
        "hoyer_scores": hoyer_scores,
        "final_scores": final_scores,
        "best_local_index": best_local
    }
    return best_global_idx, best_score, details

# -----------------------
# Main
# -----------------------
def main():
    print("Starting retrieval system (Using CSV directly)...")
    idx_title, idx_abstract, df, retrieval_model, sparse_model, sparse_tokenizer, E_corpus, Es_corpus, llm = load_resources()
    print("Ready. Enter a query (type 'exit' to quit).")

    while True:
        q = input("\nQuery> ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        if not q:
            continue

        # 1) Search
        full_query = QUERY_INSTRUCTION + q
        q_vec = retrieval_model.encode([full_query], normalize_embeddings=True).astype(np.float32)

        title_matches = query_index(idx_title, q_vec, TOP_K_CANDIDATES, TITLE_THRESHOLD)
        abs_matches = query_index(idx_abstract, q_vec, TOP_K_CANDIDATES, ABSTRACT_THRESHOLD)

        all_indices = set(title_matches.keys()) | set(abs_matches.keys())
        if not all_indices:
            print("No candidates above thresholds.")
            continue

        scored_results = []
        for idx in all_indices:
            s_t = title_matches.get(idx, -1.0)
            s_a = abs_matches.get(idx, -1.0)
            if s_t > s_a:
                scored_results.append((idx, s_t, "title"))
            else:
                scored_results.append((idx, s_a, "abstract"))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # 2) Best Match
        best_idx, best_score, best_source = scored_results[0]
        row = df.iloc[int(best_idx)]
        
        # Now we can safely access 'abstract' because we loaded the full CSV
        best_text_display = f"Title: {row.get('title','')}\nAbstract: {row.get('abstract','')}"
        best_comp_text = str(row['title']) if best_source == 'title' else str(row['abstract'])
        
        # Clean the title for deduplication
        best_title_clean = str(row.get('title', '')).strip().lower()

        print(f"\n[Best Match] ID={best_idx} score={best_score:.4f} source={best_source}")
        print("  ", str(row.get("title", ""))[:200].replace("\n", " "))

        # 3) Deduplication & Candidate Collection
        candidate_indices = []
        candidate_texts = []
        
        for idx_tuple in scored_results[1:]:
            idx = int(idx_tuple[0])
            r = df.iloc[idx]
            
            curr_title = str(r.get('title', ''))
            curr_title_clean = curr_title.strip().lower()
            
            # Deduplicate
            if curr_title_clean == best_title_clean:
                continue
            
            # Text for comparison
            candidate_indices.append(idx)
            candidate_texts.append(str(r['title']) if best_source == 'title' else str(r['abstract']))
            
            if len(candidate_indices) >= TOP_K_CANDIDATES:
                break

        # 4) Rerank
        if not candidate_indices:
            print("No candidates to rerank.")
            contra_idx, contra_score, details = None, None, {}
        else:
            print(f"Reranking {len(candidate_indices)} candidates...")
            contra_idx, contra_score, details = rerank_topk_and_find_contradiction(
                best_comp_text, best_idx, best_source,
                candidate_indices, candidate_texts,
                retrieval_model, sparse_model, sparse_tokenizer,
                E_corpus=E_corpus, Es_corpus=Es_corpus, alpha=ALPHA, device=DEVICE
            )

        if contra_idx is None:
            print("No contradiction found.")
            contra_text_display = "No contradiction found."
        else:
            r2 = df.iloc[int(contra_idx)]
            t2_text = str(r2.get('title', ''))
            a2_text = str(r2.get('abstract', ''))
            contra_text_display = f"Title: {t2_text}\nAbstract: {a2_text}"
            
            print(f"[Contradiction] ID={contra_idx} score={contra_score:.4f}")
            print("  ", t2_text[:200].replace("\n", " "))

        # 5) LLM
        if llm is not None and llm.model is not None:
            print("\nGenerating LLM answer (may take time)...")
            answer = llm.generate_response(q, best_text_display, contra_text_display)
            print("\n=== LLM ANSWER ===\n")
            print(answer)
            print("\n==================\n")
        else:
            print("\nLLM not available or not loaded. Skipping LLM step.")

if __name__ == "__main__":
    main()