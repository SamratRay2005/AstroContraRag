import faiss
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import SentenceTransformer
import sys
import os
import math

# --- Configuration ---
# 1. File Paths
TITLE_INDEX_PATH = 'arxiv_title_index.bin'
ABSTRACT_INDEX_PATH = 'arxiv_abstract_index.bin'
METADATA_PATH = 'arxiv_metadata.parquet'
SPARSE_CHECKPOINT = 'sparsecl_checkpoint.pth'

# 2. Models
RETRIEVAL_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# 3. Search Settings
TOP_K_CANDIDATES = 500 
TITLE_THRESHOLD = 0.57     
ABSTRACT_THRESHOLD = 0.71

# 4. Contradiction Hyperparameters (FROM YOUR FINDINGS)
ALPHA = 5.88454 

# 5. Instructions
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-8

# ==========================================
# SparseCL Model Definitions
# ==========================================
class SentenceEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls, p=2, dim=1)

def calculate_hoyer_vector(emb_a, emb_b):
    diff = emb_a - emb_b
    d = diff.shape[1]
    
    l1 = torch.norm(diff, p=1, dim=1)
    l2 = torch.norm(diff, p=2, dim=1) + EPS
    
    sqrt_d = math.sqrt(d)
    hoyer = (sqrt_d - (l1 / l2)) / (sqrt_d - 1.0)
    return torch.clamp(hoyer, 0.0, 1.0)

# ==========================================
# LLM Wrapper
# ==========================================
class MistralResponder:
    def __init__(self):
        print(f"Loading LLM: {LLM_MODEL_NAME}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
        except Exception as e:
            print(f"⚠️ Warning: Could not load Mistral ({e}). LLM features will be disabled.")
            self.model = None

    def generate_response(self, user_query, best_context, contra_context):
        if not self.model:
            return "LLM not loaded."

        prompt = f"""[INST] You are an expert Astronomy research assistant. 
User Query: "{user_query}"

I have retrieved two relevant scientific contexts for you.
1. The Best Match (Consensus View):
{best_context}

2. A Contradictory or Alternative View found in the literature:
{contra_context}

Task: Answer the user's query comprehensively. 
- First, explain the consensus view based on the Best Match.
- Second, explicitly highlight the contradiction or alternative methodology found in the second context.
- Discuss why this contradiction might exist (e.g., different simulation methods, data sources, or theoretical assumptions).
[/INST]"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True, 
                temperature=0.7,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in response:
            return response.split("[/INST]")[-1].strip()
        return response

# ==========================================
# Main Logic
# ==========================================
def load_resources():
    print("Loading Retrieval Resources...")
    try:
        # Load Indices
        idx_title = faiss.read_index(TITLE_INDEX_PATH)
        idx_abstract = faiss.read_index(ABSTRACT_INDEX_PATH)
        df = pd.read_parquet(METADATA_PATH)
        
        # Load Standard BGE Model (for Retrieval)
        retrieval_model = SentenceTransformer(RETRIEVAL_MODEL_NAME)
        
        # Load SparseCL Model (for Contradiction Scoring)
        print(f"Loading SparseCL Checkpoint: {SPARSE_CHECKPOINT}")
        sparse_model = SentenceEncoder(RETRIEVAL_MODEL_NAME).to(DEVICE)
        
        if os.path.exists(SPARSE_CHECKPOINT):
            ckpt = torch.load(SPARSE_CHECKPOINT, map_location=DEVICE)
            # Handle potential DataParallel wrapping or state_dict nesting
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            else:
                state_dict = ckpt
            
            new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
            sparse_model.load_state_dict(new_state)
            sparse_model.eval()
        else:
            print("❌ Error: SparseCL checkpoint not found!")
            sys.exit(1)
            
        # Load Tokenizer for SparseCL (same as BGE)
        sparse_tokenizer = AutoTokenizer.from_pretrained(RETRIEVAL_MODEL_NAME)
        
        # Load LLM
        llm = MistralResponder()
        
        return idx_title, idx_abstract, df, retrieval_model, sparse_model, sparse_tokenizer, llm
    except Exception as e:
        print(f"Error loading resources: {e}")
        sys.exit(1)

def query_index(index, query_vector, k, threshold):
    D, I = index.search(query_vector, k)
    matches = {}
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue 
        if score >= threshold:
            matches[idx] = score
    return matches

def get_contradiction_candidate(sparse_model, tokenizer, best_text, candidate_texts, candidate_indices):
    """
    Computes Score = Cosine + Alpha * Hoyer.
    Returns the index with the HIGHEST score (Highest Likelihood of Contradiction).
    """
    if not candidate_texts:
        return None, 0.0

    # Prepare inputs
    all_texts = [best_text] + candidate_texts
    
    # Tokenize
    inputs = tokenizer(all_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        embeddings = sparse_model(inputs['input_ids'], inputs['attention_mask'])
    
    best_emb = embeddings[0].unsqueeze(0) # Shape [1, D]
    cand_embs = embeddings[1:]            # Shape [N, D]
    
    # 1. Cosine Similarity
    cos_sims = torch.mm(cand_embs, best_emb.T).squeeze(1) # Shape [N]
    
    # 2. Hoyer Vector Calculation
    best_emb_expanded = best_emb.expand_as(cand_embs)
    hoyer_scores = calculate_hoyer_vector(cand_embs, best_emb_expanded) # Shape [N]
    
    # 3. Combined Score (Using your Optimal Alpha 5.88)
    # We maximize this score to find the contradiction
    final_scores = cos_sims + ALPHA * hoyer_scores
    
    # Find Index of MAXIMUM score
    max_val, max_idx_tensor = torch.max(final_scores, dim=0)
    max_idx = max_idx_tensor.item()
    
    best_contradiction_idx = candidate_indices[max_idx]
    return best_contradiction_idx, max_val.item()

def main():
    idx_title, idx_abstract, df, bge_model, sparse_model, sparse_tok, llm = load_resources()

    print("\n=======================================================")
    print(f"   ASTRONOMY CONTRADICTION RETRIEVAL SYSTEM")
    print(f"   Alpha: {ALPHA} | Accuracy Exp: ~97.8%")
    print("=======================================================\n")
    
    while True:
        query_text = input("\nEnter Query: ").strip()
        if query_text.lower() in ['exit', 'quit']:
            break
        if not query_text:
            continue

        # 1. BGE Retrieval
        full_query = QUERY_INSTRUCTION + query_text
        query_vec = bge_model.encode([full_query], normalize_embeddings=True)
        
        # Search both indices
        title_scores = query_index(idx_title, query_vec, TOP_K_CANDIDATES, TITLE_THRESHOLD)
        abstract_scores = query_index(idx_abstract, query_vec, TOP_K_CANDIDATES, ABSTRACT_THRESHOLD)
        
        all_indices = set(title_scores.keys()) | set(abstract_scores.keys())
        
        if not all_indices:
            print("No results found above thresholds.")
            continue
        
        # 2. Identify Best Match (Standard BGE Score)
        scored_results = []
        for idx in all_indices:
            s_title = title_scores.get(idx, -1.0)
            s_abs = abstract_scores.get(idx, -1.0)
            
            # Prefer Title match if scores are close, otherwise take max
            if s_title > s_abs:
                scored_results.append((idx, s_title, 'title'))
            else:
                scored_results.append((idx, s_abs, 'abstract'))
        
        # Sort desc by BGE score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        best_match = scored_results[0]
        best_idx = best_match[0]
        best_score = best_match[1]
        best_source = best_match[2] # 'title' or 'abstract'
        
        best_row = df.iloc[best_idx]
        best_text_display = f"Title: {best_row.get('title', '')}\nAbstract: {best_row.get('abstract', '')}"
        
        # This is the text used for sparse comparison (title vs title OR abstract vs abstract)
        best_comp_text = str(best_row['title']) if best_source == 'title' else str(best_row['abstract'])

        print(f"\n✅ [Best Match] ID: {best_idx} | BGE Score: {best_score:.4f}")
        print(f"Title: {best_row.get('title', 'N/A')}")

        # 3. Identify Contradictory Result
        candidate_indices = []
        candidate_texts = []
        
        # Collect candidates (skip best match)
        for res in scored_results[1:]: 
            idx = res[0]
            row = df.iloc[idx]
            # Enforce Domain Isolation: Compare Title-to-Title or Abstract-to-Abstract
            text = str(row['title']) if best_source == 'title' else str(row['abstract'])
            
            candidate_indices.append(idx)
            candidate_texts.append(text)
            
        if not candidate_indices:
            print("Not enough candidates for contradiction search.")
            contra_text_display = "No contradiction found."
        else:
            print(f"🔍 Scanning {len(candidate_indices)} candidates for contradictions...")
            
            contra_idx, contra_score = get_contradiction_candidate(
                sparse_model, sparse_tok, 
                best_comp_text, candidate_texts, candidate_indices
            )
            
            contra_row = df.iloc[contra_idx]
            contra_text_display = f"Title: {contra_row.get('title', '')}\nAbstract: {contra_row.get('abstract', '')}"
            
            print(f"\n❌ [Contradiction Found] ID: {contra_idx} | SparseCL Score: {contra_score:.4f}")
            print(f"Title: {contra_row.get('title', 'N/A')}")

        # 4. Generate LLM Response
        print("\n🤖 Generating Mistral Analysis...")
        answer = llm.generate_response(query_text, best_text_display, contra_text_display)
        
        print("\n" + "="*80)
        print("MISTRAL ANSWER:")
        print("="*80)
        print(answer)
        print("-" * 80)

if __name__ == "__main__":
    main()