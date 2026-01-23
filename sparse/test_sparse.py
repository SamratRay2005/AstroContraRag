#!/usr/bin/env python3
import csv
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np
import json
from tqdm import tqdm

# -----------------------
# Configuration
# -----------------------
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHECKPOINT_PATH = "sparsecl_checkpoint.pth" 
CSV_PATH = "dataset.csv" # Updated to new file
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
EPS = 1e-8

# -----------------------
# Model Wrapper
# -----------------------
class SentenceEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls, p=2, dim=1)

# -----------------------
# Hoyer Calculation
# -----------------------
def calculate_hoyer_vector(emb_a, emb_b):
    diff = emb_a - emb_b
    d = diff.shape[1]
    
    l1 = torch.norm(diff, p=1, dim=1)
    l2 = torch.norm(diff, p=2, dim=1) + EPS
    
    sqrt_d = math.sqrt(d)
    hoyer = (sqrt_d - (l1 / l2)) / (sqrt_d - 1.0)
    return torch.clamp(hoyer, 0.0, 1.0)

# -----------------------
# Metrics Pre-Calculation
# -----------------------
def get_dataset_metrics(model_dense, model_sparse, tokenizer, rows):
    """
    Pre-calculates Cosine and Hoyer values for all rows so we can optimize Alpha instantly.
    Handles 'contradictions' column as a JSON list.
    """
    metrics = []
    
    print(f"Pre-calculating metrics for {len(rows)} source rows (will explode to more test cases)...")
    
    with torch.no_grad():
        # Using tqdm for progress tracking
        for i, row in enumerate(tqdm(rows, desc="Encoding")):
            claim = row.get('claim', "")
            para = row.get('paraphrase', "")
            contra_json = row.get('contradictions', "[]")
            
            # Skip empty or malformed rows
            if not claim or not para:
                continue

            # Parse JSON list of contradictions
            try:
                contradictions = json.loads(contra_json)
                if not isinstance(contradictions, list):
                    continue
            except json.JSONDecodeError:
                continue

            # Pre-calculate Anchor and Paraphrase Embeddings (reuse for all contradictions)
            def get_embs(text):
                inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                d = model_dense(**inputs).last_hidden_state[:, 0, :]
                d = F.normalize(d, p=2, dim=1)
                s = model_sparse(inputs['input_ids'], inputs['attention_mask'])
                return d, s

            d_claim, s_claim = get_embs(claim)
            d_para, s_para = get_embs(para)
            
            # Calculate Paraphrase Metrics once
            cos_para = torch.mm(d_claim, d_para.T).item()
            h_para = calculate_hoyer_vector(s_claim, s_para).item()

            # Iterate through ALL contradictions in the list
            for c_idx, contra_text in enumerate(contradictions):
                # Filter out garbage short strings
                if not isinstance(contra_text, str) or len(contra_text.strip()) < 5:
                    continue

                d_contra, s_contra = get_embs(contra_text)
                
                cos_contra = torch.mm(d_claim, d_contra.T).item()
                h_contra = calculate_hoyer_vector(s_claim, s_contra).item()
                
                metrics.append({
                    'id': f"{i}_{c_idx}", # Unique ID combining row and index
                    'cos_p': cos_para,
                    'cos_c': cos_contra,
                    'h_p': h_para,
                    'h_c': h_contra,
                    'claim': claim
                })
            
    print(f"Generated {len(metrics)} total evaluation pairs.")
    return metrics

# -----------------------
# Exact Global Search Strategy
# -----------------------
def find_best_alpha(metrics):
    """
    Scans for the alpha that maximizes Contradiction Retrieval Accuracy.
    """
    candidates = {0.0} 
    
    # 1. Collect all candidate alphas where the decision flips
    for m in metrics:
        numerator = m['cos_c'] - m['cos_p']
        denominator = m['h_p'] - m['h_c']
        
        if abs(denominator) > 1e-9:
            critical_alpha = numerator / denominator
            if critical_alpha > 0:
                # Add the point exactly and slightly after to break ties
                candidates.add(critical_alpha)
                candidates.add(critical_alpha + 1e-5)
    
    # Add standard steps
    for x in np.arange(0, 5.0, 0.1):
        candidates.add(x)
        
    sorted_alphas = sorted(list(candidates))
    print(f"\n--- Scanning {len(sorted_alphas)} potential optimal values ---")
    
    best_acc = -1
    best_alpha = 0
    
    # 2. Sweep
    for alpha in sorted_alphas:
        correct = 0
        total = len(metrics)
        
        for m in metrics:
            score_p = m['cos_p'] + alpha * m['h_p']
            score_c = m['cos_c'] + alpha * m['h_c']
            
            # We want Contradiction Score > Paraphrase Score
            if score_c > score_p:
                correct += 1
        
        acc = correct / total
        
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
    
    return best_alpha, best_acc

# -----------------------
# Main
# -----------------------
def main():
    # 1. Setup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Loading Dense Model...")
    model_dense = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model_dense.eval()
    
    print("Loading Sparse Model...")
    model_sparse = SentenceEncoder(MODEL_NAME).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if 'model_state_dict' in ckpt:
            new_state = {k.replace("module.", ""): v for k, v in ckpt['model_state_dict'].items()}
            model_sparse.load_state_dict(new_state)
        else:
            model_sparse.load_state_dict(ckpt)
    else:
        print("Warning: Checkpoint not found.")
    model_sparse.eval()

    # 2. Read Data
    rows = []
    if os.path.exists(CSV_PATH):
        # Increase field size limit for large CSVs
        csv.field_size_limit(10**7)
        with open(CSV_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader: rows.append(r)
    else:
        print(f"CSV file {CSV_PATH} not found.")
        return

    # 3. Get Raw Numbers
    metrics = get_dataset_metrics(model_dense, model_sparse, tokenizer, rows)
    
    if not metrics:
        print("No metrics calculated. Check dataset format.")
        return

    # 4. Find Best Alpha
    print("\n--- Starting Optimization ---")
    best_alpha, best_acc = find_best_alpha(metrics)
    
    print("\n" + "="*40)
    print(f"OPTIMAL ALPHA FOUND: {best_alpha:.5f}")
    print(f"MAX ACCURACY:        {best_acc:.2%}")
    print("="*40)
    
    # 5. Show Validation on Best Alpha
    print("\n--- Verifying Best Result (Sample) ---")
    correct = 0
    fail_count = 0
    
    for m in metrics:
        score_p = m['cos_p'] + best_alpha * m['h_p']
        score_c = m['cos_c'] + best_alpha * m['h_c']
        
        verdict = "PASS" if score_c > score_p else "FAIL"
        
        if verdict == "PASS": correct += 1
        
        if verdict == "FAIL":
            fail_count += 1
            # Only print first 10 failures to avoid spam
            if fail_count <= 10:
                print(f"[FAIL] ID {m['id']}: CosP={m['cos_p']:.3f} CosC={m['cos_c']:.3f} | HoyerP={m['h_p']:.3f} HoyerC={m['h_c']:.3f}")
                print(f"       FinalP={score_p:.3f} FinalC={score_c:.3f} (Contradiction was lower)")

    print(f"\nFinal Verified Count: {correct}/{len(metrics)}")

if __name__ == "__main__":
    main()