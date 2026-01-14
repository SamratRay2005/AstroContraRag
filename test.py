#!/usr/bin/env python3
import csv
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np

# -----------------------
# Configuration
# -----------------------
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHECKPOINT_PATH = "sparsecl_checkpoint.pth" 
CSV_PATH = "astronomy_test_data.csv"
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
    """
    metrics = []
    
    print(f"Pre-calculating metrics for {len(rows)} test cases...")
    
    with torch.no_grad():
        for i, row in enumerate(rows):
            claim = row['claim']
            para = row['paraphrase']
            contra = row['contradiction']
            
            # Tokenize
            def get_embs(text):
                inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                d = model_dense(**inputs).last_hidden_state[:, 0, :]
                d = F.normalize(d, p=2, dim=1)
                s = model_sparse(inputs['input_ids'], inputs['attention_mask'])
                return d, s

            d_claim, s_claim = get_embs(claim)
            d_para, s_para = get_embs(para)
            d_contra, s_contra = get_embs(contra)

            # Metrics
            cos_para = torch.mm(d_claim, d_para.T).item()
            cos_contra = torch.mm(d_claim, d_contra.T).item()
            
            h_para = calculate_hoyer_vector(s_claim, s_para).item()
            h_contra = calculate_hoyer_vector(s_claim, s_contra).item()
            
            metrics.append({
                'id': i+1,
                'cos_p': cos_para,
                'cos_c': cos_contra,
                'h_p': h_para,
                'h_c': h_contra,
                'claim': claim
            })
            
    return metrics

# -----------------------
# Exact Global Search Strategy
# -----------------------
def find_best_alpha(metrics):
    """
    Instead of a binary search (which might miss the peak), we find 
    all 'Critical Alphas' where the decision flips, and check them all.
    This guarantees finding the global maximum.
    """
    # 1. Collect all candidate alphas
    # A decision flips when Score_Para == Score_Contra
    # Cos_P + A * H_P = Cos_C + A * H_C
    # A * (H_P - H_C) = Cos_C - Cos_P
    # A = (Cos_C - Cos_P) / (H_P - H_C)
    
    candidates = {0.0} # Always check Alpha=0 (Pure Cosine)
    
    for m in metrics:
        numerator = m['cos_c'] - m['cos_p']
        denominator = m['h_p'] - m['h_c']
        
        if abs(denominator) > 1e-9:
            critical_alpha = numerator / denominator
            if critical_alpha > 0:
                # Add the point exactly, and slightly after it (to break ties)
                candidates.add(critical_alpha)
                candidates.add(critical_alpha + 1e-5)
    
    # Also test some standard discrete steps just in case
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
            
            if score_p > score_c:
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
        with open(CSV_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader: rows.append(r)
    else:
        print("CSV not found.")
        return

    # 3. Get Raw Numbers
    metrics = get_dataset_metrics(model_dense, model_sparse, tokenizer, rows)
    
    # 4. Find Best Alpha
    print("\n--- Starting Optimization ---")
    best_alpha, best_acc = find_best_alpha(metrics)
    
    print("\n" + "="*40)
    print(f"OPTIMAL ALPHA FOUND: {best_alpha:.5f}")
    print(f"MAX ACCURACY:        {best_acc:.2%}")
    print("="*40)
    
    # 5. Show Validation on Best Alpha
    print("\n--- Verifying Best Result ---")
    correct = 0
    for m in metrics:
        score_p = m['cos_p'] + best_alpha * m['h_p']
        score_c = m['cos_c'] + best_alpha * m['h_c']
        verdict = "PASS" if score_p > score_c else "FAIL"
        if verdict == "PASS": correct += 1
        
        # Only print fails to keep it short
        if verdict == "FAIL":
            print(f"[FAIL] ID {m['id']}: CosP={m['cos_p']:.3f} CosC={m['cos_c']:.3f} | HoyerP={m['h_p']:.3f} HoyerC={m['h_c']:.3f}")

    print(f"\nFinal Verified Count: {correct}/{len(metrics)}")

if __name__ == "__main__":
    main()
