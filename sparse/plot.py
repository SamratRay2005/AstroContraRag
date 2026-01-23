#!/usr/bin/env python3
# FAST SparseCL diagnostics (batched + GPU)
import os, math, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# -----------------------
# Config
# -----------------------
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHECKPOINT_PATH = "/kaggle/input/rssssssss/pytorch/default/1/sparsecl_checkpoint.pth"
CSV_PATH = "/kaggle/input/rssssss/dataset.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

BATCH_SIZE = 64          # try 32/64/128 depending on GPU RAM
MAX_LEN = 256            # keep smaller for speed (paper uses 256 for some datasets)
USE_FP16 = True          # only works on CUDA


# -----------------------
# Models
# -----------------------
class SentenceEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(cls, p=2, dim=1)


def calculate_hoyer_batch(emb_a, emb_b):
    """
    emb_a, emb_b: (N, D)
    returns: (N,) hoyer in [0,1]
    """
    diff = emb_a - emb_b
    d = diff.shape[1]

    l1 = torch.norm(diff, p=1, dim=1)
    l2 = torch.norm(diff, p=2, dim=1) + EPS

    sqrt_d = math.sqrt(d)
    hoyer = (sqrt_d - (l1 / l2)) / (sqrt_d - 1.0)
    return torch.clamp(hoyer, 0.0, 1.0)


@torch.inference_mode()
def embed_texts(texts, tokenizer, model_dense, model_sparse, batch_size=BATCH_SIZE):
    """
    Batched embedding for both dense + sparse models.
    Returns:
        dense_embs: (N, D)
        sparse_embs: (N, D)
    """
    dense_chunks = []
    sparse_chunks = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inp = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        d = model_dense(**inp).last_hidden_state[:, 0, :]
        d = F.normalize(d, p=2, dim=1)

        s = model_sparse(inp["input_ids"], inp["attention_mask"])  # already normalized
        dense_chunks.append(d)
        sparse_chunks.append(s)

    return torch.cat(dense_chunks, dim=0), torch.cat(sparse_chunks, dim=0)


def main():
    print("DEVICE:", DEVICE)

    # Speed hint for matmul on newer GPUs
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading dense model...")
    model_dense = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    print("Loading sparse model...")
    model_sparse = SentenceEncoder(MODEL_NAME).to(DEVICE).eval()

    # Load SparseCL checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print("Loading checkpoint:", CHECKPOINT_PATH)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model_sparse.load_state_dict(state_dict, strict=False)
    else:
        print("⚠️ Checkpoint not found, using base weights.")

    # FP16 for speed (CUDA only)
    if USE_FP16 and DEVICE.type == "cuda":
        model_dense = model_dense.half()
        model_sparse = model_sparse.half()
        print("Using FP16 inference.")

    # Load CSV
    if not os.path.exists(CSV_PATH):
        print("CSV not found:", CSV_PATH)
        return

    data = pd.read_csv(CSV_PATH)
    print("Rows:", len(data))

    # ---------------------------------------------------------
    # Build pairs list: (type, claim_text, other_text)
    # ---------------------------------------------------------
    pair_types = []
    claim_texts = []
    other_texts = []

    # faster than iterrows()
    for row in tqdm(data.itertuples(index=False), total=len(data), desc="Reading rows"):
        try:
            claim = getattr(row, "claim", "")
            para = getattr(row, "paraphrase", "")
            contra_json = getattr(row, "contradictions", "[]")

            if not isinstance(claim, str) or not isinstance(para, str):
                continue
            claim = claim.strip()
            para = para.strip()
            if len(claim) < 3 or len(para) < 3:
                continue

            # Paraphrase pair
            pair_types.append("Paraphrase")
            claim_texts.append(claim)
            other_texts.append(para)

            # Contradictions list
            if isinstance(contra_json, str):
                contradictions = json.loads(contra_json)
            elif isinstance(contra_json, list):
                contradictions = contra_json
            else:
                contradictions = []

            for c in contradictions:
                if not isinstance(c, str):
                    continue
                c = c.strip()
                if len(c) < 5:
                    continue
                pair_types.append("Contradiction")
                claim_texts.append(claim)
                other_texts.append(c)

        except Exception:
            continue

    if not pair_types:
        print("No valid pairs found.")
        return

    print(f"Total pairs: {len(pair_types)}")

    # ---------------------------------------------------------
    # Embed ALL claims and ALL others in big batches
    # ---------------------------------------------------------
    print("Embedding claims...")
    d_claim, s_claim = embed_texts(claim_texts, tokenizer, model_dense, model_sparse)

    print("Embedding paired texts...")
    d_other, s_other = embed_texts(other_texts, tokenizer, model_dense, model_sparse)

    # ---------------------------------------------------------
    # Compute metrics (vectorized)
    # ---------------------------------------------------------
    # cosine for normalized vectors = dot product
    cos_scores = (d_claim * d_other).sum(dim=1).float().cpu().numpy()

    hoyer_scores = calculate_hoyer_batch(s_claim.float(), s_other.float()).cpu().numpy()

    df_res = pd.DataFrame({
        "Type": pair_types,
        "Cosine": cos_scores,
        "Hoyer": hoyer_scores
    })

    print("Generated metrics:", len(df_res))

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(
        data=df_res,
        x="Hoyer",
        hue="Type",
        kde=True,
        bins=30,
        ax=axes[0]
    )
    axes[0].set_title("Hoyer Sparsity Distribution (Fast)")
    axes[0].set_xlabel("Hoyer Sparsity Score")

    sns.scatterplot(
        data=df_res,
        x="Cosine",
        y="Hoyer",
        hue="Type",
        style="Type",
        alpha=0.5,
        ax=axes[1]
    )
    axes[1].set_title("Cosine vs Hoyer Separation (Fast)")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Hoyer Sparsity")

    plt.tight_layout()
    plt.savefig("sparsecl_diagnosis_fast.png", dpi=200)
    print("✅ Saved: sparsecl_diagnosis_fast.png")


if __name__ == "__main__":
    main()
