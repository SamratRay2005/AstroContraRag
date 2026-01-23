import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import json
import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    MODEL_NAME = "BAAI/bge-base-en-v1.5"
    MAX_LEN = 128
    BATCH_SIZE = 16 
    EPOCHS = 3
    LR = 2e-5
    TEMPERATURE = 0.05
    DATA_PATH = "dataset.csv" 
    CHECKPOINT_PATH = "sparsecl_checkpoint.pth"
    SAVE_STEPS = 500
    
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 2. Robust Dataset (Pandas-based)
# ==========================================
class ExplodedCSVDataset(Dataset):
    def __init__(self, data_path):
        print(f"--- LOADING DATASET ({data_path}) ---")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} not found!")

        # 1. Load data into RAM using Pandas (Handles multiline CSVs correctly)
        df = pd.read_csv(data_path)
        
        # 2. Explode the dataset
        self.samples = []
        
        print(f"Parsing {len(df)} rows...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Basic fields
                claim = row['claim']
                paraphrase = row['paraphrase']
                
                # Parse JSON
                # Handle potential string/object discrepancies
                contra_raw = row['contradictions']
                if isinstance(contra_raw, str):
                    contradictions = json.loads(contra_raw)
                else:
                    contradictions = contra_raw # Already list if pandas inferred it (rare for JSON)
                
                if not isinstance(contradictions, list):
                    continue

                # Create a sample for EACH valid contradiction
                for contra in contradictions:
                    if contra and isinstance(contra, str) and len(contra.strip()) > 5:
                        self.samples.append({
                            'anchor': claim,
                            'positive': contra,
                            'negative': paraphrase
                        })
                        
            except Exception as e:
                # Skip malformed rows
                continue

        print(f"✅ Loaded {len(self.samples)} training samples.")
        print("-------------------------------------------\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    # Filter Nones just in case
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    return {
        'anchor': [item['anchor'] for item in batch],
        'positive': [item['positive'] for item in batch],
        'negative': [item['negative'] for item in batch]
    }

# ==========================================
# 3. Model & Loss (Optimized)
# ==========================================
class HoyerSparsityLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(HoyerSparsityLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = 1e-8

    def forward(self, anchor_emb, pos_emb, neg_emb):
        # anchor: [B, D], pos: [B, D], neg: [B, D]
        
        # We need to broadcast to compute pairwise differences in the batch
        # A: [B, 1, D]
        # P: [1, B, D]
        # N: [1, B, D]
        a_ex = anchor_emb.unsqueeze(1)
        p_ex = pos_emb.unsqueeze(0)
        n_ex = neg_emb.unsqueeze(0)
        
        d = anchor_emb.shape[1]
        sqrt_d = np.sqrt(d)

        # Helper to compute Hoyer Score matrix
        def compute_hoyer_matrix(expanded_anchor, expanded_target):
            # Diff: [B, B, D]
            diff = expanded_anchor - expanded_target
            l1 = torch.norm(diff, p=1, dim=2)
            l2 = torch.norm(diff, p=2, dim=2) + self.epsilon
            # Hoyer Formula: (sqrt(d) - l1/l2) / (sqrt(d) - 1)
            return (sqrt_d - (l1/l2)) / (sqrt_d - 1)

        # 1. Compute Hoyer Scores
        # matrix_pos[i, j] = Hoyer(Anchor_i, Positive_j)
        matrix_pos = compute_hoyer_matrix(a_ex, p_ex)
        
        # matrix_neg[i, j] = Hoyer(Anchor_i, Negative_j)
        matrix_neg = compute_hoyer_matrix(a_ex, n_ex)
        
        # 2. Contrastive Loss (InfoNCE)
        # We want to MAXIMIZE the Diagonal of matrix_pos (Anchor_i vs Positive_i)
        
        # Numerator: exp(Hoyer(A_i, P_i) / T)
        # We extract the diagonal for the numerator
        pos_diag = torch.diag(matrix_pos)
        numerator = torch.exp(pos_diag / self.temperature)
        
        # Denominator: Sum of exps of ALL pairs (Positives + Negatives)
        # This treats other batch items as "In-Batch Negatives"
        denom_pos = torch.sum(torch.exp(matrix_pos / self.temperature), dim=1)
        denom_neg = torch.sum(torch.exp(matrix_neg / self.temperature), dim=1)
        
        # Standard LogSoftmax form
        loss = -torch.log(numerator / (denom_pos + denom_neg))
        
        return loss.mean()

class SentenceEncoder(nn.Module):
    def __init__(self, model_name):
        super(SentenceEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 1. Get token embeddings
        token_embeddings = outputs.last_hidden_state # [Batch, SeqLen, Dim]
        
        # 2. Create mask for Mean Pooling (exclude padding tokens)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # 3. Sum and Divide
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        mean_emb = sum_embeddings / sum_mask
        
        # 4. Normalize (Critical for Cosine Similarity later)
        return F.normalize(mean_emb, p=2, dim=1)

# ==========================================
# 4. Training Loop
# ==========================================
def save_checkpoint(model, optimizer, epoch, step, loss):
    print(f"\nSaving checkpoint at Epoch {epoch} Step {step}...")
    state = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, Config.CHECKPOINT_PATH)

def load_checkpoint(model, optimizer):
    if os.path.exists(Config.CHECKPOINT_PATH):
        print("Checkpoint found! Loading...")
        checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['step']
    return 0, 0

def train():
    set_seed(Config.SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = SentenceEncoder(Config.MODEL_NAME).to(Config.DEVICE)
    
    # Init Dataset
    try:
        dataset = ExplodedCSVDataset(Config.DATA_PATH)
    except Exception as e:
        print(f"Fatal Error loading data: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    criterion = HoyerSparsityLoss(temperature=Config.TEMPERATURE)
    optimizer = AdamW(model.parameters(), lr=Config.LR)
    
    start_epoch, start_step = load_checkpoint(model, optimizer)
    current_loss_val = 0.0 
    
    print(f"Starting training on {Config.DEVICE}...")
    model.train()
    
    for epoch in range(start_epoch, Config.EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for batch_idx, batch in enumerate(loop):
            if epoch == start_epoch and batch_idx < start_step:
                continue
            
            if not batch: continue
            
            # Tokenize
            def tok(texts):
                return tokenizer(texts, padding=True, truncation=True, max_length=Config.MAX_LEN, return_tensors='pt').to(Config.DEVICE)
            
            anchors_tok = tok(batch['anchor'])
            pos_tok = tok(batch['positive'])
            neg_tok = tok(batch['negative'])
            
            optimizer.zero_grad()
            
            anchor_emb = model(anchors_tok['input_ids'], anchors_tok['attention_mask'])
            pos_emb = model(pos_tok['input_ids'], pos_tok['attention_mask'])
            neg_emb = model(neg_tok['input_ids'], neg_tok['attention_mask'])
            
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()
            
            current_loss_val = loss.item()
            loop.set_postfix(loss=f"{current_loss_val:.4f}")
            
            if batch_idx > 0 and batch_idx % Config.SAVE_STEPS == 0:
                save_checkpoint(model, optimizer, epoch, batch_idx, current_loss_val)

        start_step = 0
        save_checkpoint(model, optimizer, epoch + 1, 0, current_loss_val)
        print(f"Epoch {epoch+1} completed.")

if __name__ == "__main__":
    train()