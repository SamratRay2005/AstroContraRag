import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import glob
import sys
import random

# --- CONFIGURATION ---
MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Files
TITLE_INDEX_PATH = 'arxiv_title_index.bin'
ABSTRACT_INDEX_PATH = 'arxiv_abstract_index.bin'
METADATA_PATH = 'arxiv_metadata.parquet'
TRAIN_FILE = "train.csv.gz"

# Tuning Grid
INITIAL_SEARCH_K = 200 
# Testing a wide range since we expect mixed results
THRESHOLDS_TO_TEST = [round(x, 2) for x in np.arange(0.35, 0.90, 0.02)]

# We will generate a new column for our test
COL_QUERY = 'simulated_query'         
COL_GROUND_TRUTH = 'arxiv_id'

def perturb_text(text):
    """
    Simulates a fuzzy user query by removing random words.
    """
    words = str(text).split()
    if len(words) <= 3:
        return text # Keep short titles as is
    
    # Randomly keep 50% to 80% of the words
    keep_ratio = random.uniform(0.5, 0.8)
    num_keep = int(len(words) * keep_ratio)
    
    # Select random words but keep them in original order (usually helps semantics)
    selected_words = sorted(random.sample(list(enumerate(words)), num_keep), key=lambda x: x[0])
    return " ".join([word for idx, word in selected_words])

def load_and_perturb_data():
    print("Loading resources...")
    model = SentenceTransformer(MODEL_NAME)
    
    idx_title = faiss.read_index(TITLE_INDEX_PATH)
    idx_abstract = faiss.read_index(ABSTRACT_INDEX_PATH)
    
    print(f"Loading metadata from {METADATA_PATH}...")
    meta_df = pd.read_parquet(METADATA_PATH)
    real_ids = meta_df['arxiv_id'].tolist()
    
    print(f"Loading dataset: {TRAIN_FILE}...")
    try:
        train_df = pd.read_csv(TRAIN_FILE, compression='gzip', nrows=10000, low_memory=False)
    except Exception as e:
        print(f"Error reading {TRAIN_FILE}: {e}")
        sys.exit(1)
            
    # Sample 500 rows
    sample_size = min(500, len(train_df))
    print(f"Loaded {len(train_df)} rows. Sampling {sample_size} for mixed testing...")
    train_sample = train_df.sample(n=sample_size, random_state=42).copy()
    
    # --- SIMULATION LOGIC ---
    print("\n--- Simulating User Queries ---")
    new_queries = []
    perturb_count = 0
    
    for title in train_sample['title']:
        if random.random() > 0.5:
            # 50% Chance: Messy Query (remove words)
            new_queries.append(perturb_text(title))
            perturb_count += 1
        else:
            # 50% Chance: Perfect Query
            new_queries.append(title)
            
    train_sample['simulated_query'] = new_queries
    print(f"Dataset prepared: {perturb_count} fuzzy queries / {sample_size - perturb_count} exact queries.")
    print(f"Example Fuzzy Query: '{new_queries[0]}'")
    
    return model, idx_title, idx_abstract, real_ids, train_sample

def evaluate_thresholds(model, idx_title, idx_abstract, real_ids, train_df):
    results_grid = []

    print("\n--- Encoding Queries ---")
    queries = [QUERY_INSTRUCTION + str(q) for q in train_df[COL_QUERY].tolist()]
    query_vecs = model.encode(queries, normalize_embeddings=True, show_progress_bar=True)
    
    ground_truths = train_df[COL_GROUND_TRUTH].tolist()

    print(f"\n--- Running Initial Search (Top {INITIAL_SEARCH_K}) ---")
    D_title, I_title = idx_title.search(query_vecs, INITIAL_SEARCH_K)
    D_abs, I_abs = idx_abstract.search(query_vecs, INITIAL_SEARCH_K)

    print(f"\n--- Testing {len(THRESHOLDS_TO_TEST)**2} Combinations ---")
    
    total_queries = len(queries)

    for t_thresh in THRESHOLDS_TO_TEST:
        for a_thresh in THRESHOLDS_TO_TEST:
            
            total_recall_hits = 0
            total_retrieved_count = 0
            
            for i in range(total_queries):
                correct_id = str(ground_truths[i]).strip()
                
                # Title matches
                t_mask = D_title[i] >= t_thresh
                t_matches = I_title[i][t_mask]
                
                # Abstract matches
                a_mask = D_abs[i] >= a_thresh
                a_matches = I_abs[i][a_mask]
                
                # Union
                union_indices = set(t_matches) | set(a_matches)
                union_indices.discard(-1)
                
                count = len(union_indices)
                total_retrieved_count += count
                
                if count > 0:
                    found = False
                    for idx in union_indices:
                        if idx < len(real_ids):
                            if str(real_ids[idx]).strip() == correct_id:
                                found = True
                                break
                    if found:
                        total_recall_hits += 1
            
            avg_recall = (total_recall_hits / total_queries) * 100
            avg_count = total_retrieved_count / total_queries
            
            results_grid.append({
                'Title_Thresh': t_thresh,
                'Abs_Thresh': a_thresh,
                'Recall (%)': avg_recall,
                'Avg_Results': avg_count
            })
            
            if len(results_grid) % 100 == 0:
                print(f"Tested {len(results_grid)} combos...")

    return pd.DataFrame(results_grid)

def main():
    try:
        model, idx_title, idx_abstract, real_ids, train_sample = load_and_perturb_data()
        
        df_results = evaluate_thresholds(model, idx_title, idx_abstract, real_ids, train_sample)
        
        print("\n--- Tuning Results (Top 10 by Recall, then Efficiency) ---")
        # We value Recall first, then low result count
        df_results = df_results.sort_values(by=['Recall (%)', 'Avg_Results'], ascending=[False, True])
        
        print(df_results.head(10).to_string(index=False))
        
        best = df_results.iloc[0]
        print(f"\n>>> SUGGESTED SETTINGS <<<")
        print(f"TITLE_THRESHOLD = {best['Title_Thresh']}")
        print(f"ABSTRACT_THRESHOLD = {best['Abs_Thresh']}")
        print(f"Stats: {best['Recall (%)']:.2f}% Recall with approx {best['Avg_Results']:.1f} results/query")
        
        df_results.to_csv("hyperparameter_mixed_results.csv", index=False)
        print("\nSaved full results to 'hyperparameter_mixed_results.csv'")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()