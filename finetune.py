import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import json
import numpy as np
import os
import csv
import random
from tqdm import tqdm
import time

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
    DATA_PATH = "dataset.csv"  # Ensure this matches your actual filename
    CHECKPOINT_PATH = "sparsecl_checkpoint.pth"
    SAVE_STEPS = 500  # Increased slightly since dataset is now larger
    
    # Auto-detect hardware
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 2. Exploded Dataset (Utilizes ALL Contradictions)
# ==========================================
class ExplodedCSVDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        # Each item in self.indices is a tuple: (file_offset_bytes, contradiction_index_in_json)
        self.indices = [] 
        self.col_map = {}
        
        print(f"--- INDEXING DATASET ({data_path}) ---")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} not found!")

        start_time = time.time()
        
        with open(data_path, 'rb') as f:
            # 1. Parse Header
            header_bytes = f.readline()
            header_str = header_bytes.decode('utf-8').replace('\ufeff', '')
            headers = next(csv.reader([header_str]))
            self.col_map = {name.strip(): i for i, name in enumerate(headers)}
            
            # Verify columns
            if 'contradictions' not in self.col_map:
                raise ValueError("CSV missing 'contradictions' column")

            col_idx_contra = self.col_map['contradictions']
            col_idx_status = self.col_map.get('status_overall', -1)

            # 2. Scan and Explode
            row_count = 0
            while True:
                offset = f.tell()
                line_bytes = f.readline()
                if not line_bytes:
                    break
                
                row_count += 1
                try:
                    line_str = line_bytes.decode('utf-8')
                    row_values = next(csv.reader([line_str]))
                    
                    # Safety check for truncated rows
                    if len(row_values) <= col_idx_contra:
                        continue

                    # Check Status (Optional, based on your logic)
                    if col_idx_status != -1:
                        status = row_values[col_idx_status]
                        if "invalid" in status:
                            continue

                    # Parse JSON Contradictions
                    contradictions_json = row_values[col_idx_contra]
                    contradictions = json.loads(contradictions_json)

                    if isinstance(contradictions, list):
                        # Add an index for EVERY valid contradiction in the list
                        for i, contra in enumerate(contradictions):
                            # Ensure it's not an empty string (failed generation)
                            if contra and isinstance(contra, str) and len(contra.strip()) > 5:
                                self.indices.append((offset, i))
                                
                except (json.JSONDecodeError, IndexError, ValueError):
                    # Skip malformed lines silently during indexing
                    continue

        elapsed = time.time() - start_time
        print(f"Scanned {row_count} raw CSV rows.")
        print(f"Created {len(self.indices)} training samples (Exploded View).")
        print(f"Indexing took {elapsed:.2f} seconds.")
        print("-------------------------------------------\n")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        offset, contra_idx = self.indices[idx]
        
        try:
            with open(self.data_path, 'rb') as f:
                f.seek(offset)
                line_bytes = f.readline()

            line_str = line_bytes.decode('utf-8')
            row_values = next(csv.reader([line_str]))
            
            # Extract fields
            claim = row_values[self.col_map['claim']]
            paraphrase = row_values[self.col_map['paraphrase']]
            
            # Get the SPECIFIC contradiction for this sample
            contradictions = json.loads(row_values[self.col_map['contradictions']])
            target_contradiction = contradictions[contra_idx]

            return {
                'anchor': claim,
                'positive': target_contradiction, # The specific Type A or Type B
                'negative': paraphrase
            }

        except Exception as e:
            print(f"[WARN] Error loading index {idx}: {e}")
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    return {
        'anchor': [item['anchor'] for item in batch],
        'positive': [item['positive'] for item in batch],
        'negative': [item['negative'] for item in batch]
    }

# ==========================================
# 3. Model & Loss (Standard Hoyer)
# ==========================================
class HoyerSparsityLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(HoyerSparsityLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = 1e-8

    def forward(self, anchor_emb, pos_emb, neg_emb):
        # Calculate Hoyer matrix elements for Positive pairs
        a_ex = anchor_emb.unsqueeze(1)
        p_ex = pos_emb.unsqueeze(0)
        n_ex = neg_emb.unsqueeze(0)
        
        d = anchor_emb.shape[1]
        sqrt_d = np.sqrt(d)

        # Diff Positive
        diff_pos = a_ex - p_ex
        l1_pos = torch.norm(diff_pos, p=1, dim=2)
        l2_pos = torch.norm(diff_pos, p=2, dim=2) + self.epsilon
        matrix_pos = (sqrt_d - (l1_pos/l2_pos)) / (sqrt_d - 1)

        # Diff Negative
        diff_neg = a_ex - n_ex
        l1_neg = torch.norm(diff_neg, p=1, dim=2)
        l2_neg = torch.norm(diff_neg, p=2, dim=2) + self.epsilon
        matrix_neg = (sqrt_d - (l1_neg/l2_neg)) / (sqrt_d - 1)
        
        # Contrastive LogSumExp
        numerator = torch.exp(torch.diag(matrix_pos) / self.temperature)
        denom_pos = torch.sum(torch.exp(matrix_pos / self.temperature), dim=1)
        denom_neg = torch.sum(torch.exp(matrix_neg / self.temperature), dim=1)
        
        loss = -torch.log(numerator / (denom_pos + denom_neg))
        return loss.mean()

class SentenceEncoder(nn.Module):
    def __init__(self, model_name):
        super(SentenceEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling is often better for BGE than CLS, but CLS is fine too. 
        # Using CLS token here as per your previous code.
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls_emb, p=2, dim=1)

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
    
    # Init Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Init Model
    model = SentenceEncoder(Config.MODEL_NAME).to(Config.DEVICE)
    
    # Init Dataset (Exploded)
    try:
        dataset = ExplodedCSVDataset(Config.DATA_PATH)
    except Exception as e:
        print(f"Dataset Init Fatal Error: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    criterion = HoyerSparsityLoss(temperature=Config.TEMPERATURE)
    optimizer = AdamW(model.parameters(), lr=Config.LR)
    
    start_epoch, start_step = load_checkpoint(model, optimizer)
    current_loss_val = 0.0 
    
    print(f"Starting training on {Config.DEVICE}...")
    print(f"Total Samples: {len(dataset)} | Batches per Epoch: {len(dataloader)}")
    model.train()
    
    for epoch in range(start_epoch, Config.EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for batch_idx, batch in enumerate(loop):
            # Resume logic
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
            
            # Save Checkpoint
            if batch_idx > 0 and batch_idx % Config.SAVE_STEPS == 0:
                save_checkpoint(model, optimizer, epoch, batch_idx, current_loss_val)

        # End of Epoch
        start_step = 0
        save_checkpoint(model, optimizer, epoch + 1, 0, current_loss_val)
        print(f"Epoch {epoch+1} completed.")

if __name__ == "__main__":
    train()