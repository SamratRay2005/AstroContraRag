import os
import sys
import pandas as pd
import csv
from tqdm import tqdm
import run  # Imports your existing run.py

# --- CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" 

INPUT_FILENAME = "finaltest.csv"
INTERMEDIATE_FILENAME = "intermediate_candidates.csv"

def main():
    # 1. Load Retrieval Resources
    print(f"Loading retrieval resources...")
    idx_title, idx_abstract, df_corpus, retrieval_model, sparse_model, sparse_tokenizer, E_corpus, Es_corpus, _ = run.load_resources()

    # 2. Load Queries
    print(f"Reading queries from {INPUT_FILENAME}...")
    try:
        df_queries = pd.read_csv(INPUT_FILENAME)
        if 'query_id' not in df_queries.columns:
            df_queries['query_id'] = df_queries.index
    except Exception as e:
        print(f"Error reading {INPUT_FILENAME}: {e}")
        return

    # Check for existing progress
    processed_ids = set()
    write_header = True
    if os.path.exists(INTERMEDIATE_FILENAME):
        try:
            existing = pd.read_csv(INTERMEDIATE_FILENAME)
            # Simple check to see if we need to restart due to column changes
            if "best_match_abstract" in existing.columns:
                processed_ids = set(existing['query_id'].unique())
                write_header = False
                print(f"Resuming: {len(processed_ids)} queries already retrieved.")
            else:
                print("Old CSV format detected. Starting fresh to ensure Titles/Abstracts are saved.")
        except:
            pass

    # Open CSV for writing
    with open(INTERMEDIATE_FILENAME, mode='a' if not write_header else 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        
        # --- UPDATED HEADER ---
        if write_header:
            writer.writerow([
                "query_id", 
                "query_text", 
                "best_match_id", 
                "best_match_title",     # New
                "best_match_abstract",  # New
                "contra_match_id", 
                "contra_match_title",   # New
                "contra_match_abstract",# New
                "contra_score", 
                "status"
            ])

        # 3. Encode Queries
        print("Encoding queries...")
        query_texts = []
        valid_rows = []
        
        for i, row in df_queries.iterrows():
            if row['query_id'] in processed_ids: continue
            
            text = str(row.get('title', ''))
            if len(text) < 10: text = str(row.get('abstract', ''))
            
            if len(text) >= 5:
                query_texts.append(run.QUERY_INSTRUCTION + text)
                valid_rows.append(row)

        if not query_texts:
            print("No new queries to process.")
            return

        all_q_vecs = retrieval_model.encode(
            query_texts, 
            batch_size=128, 
            show_progress_bar=True, 
            normalize_embeddings=True
        ).astype("float32")

        # 4. Retrieval Loop
        print("Running Retrieval & Reranking...")
        
        for matrix_idx, row in tqdm(enumerate(valid_rows), total=len(valid_rows)):
            q_id = row['query_id']
            q_vec = all_q_vecs[matrix_idx].reshape(1, -1)
            
            # --- FAISS Search ---
            title_matches = run.query_index(idx_title, q_vec, run.TOP_K_CANDIDATES, run.TITLE_THRESHOLD)
            abs_matches = run.query_index(idx_abstract, q_vec, run.TOP_K_CANDIDATES, run.ABSTRACT_THRESHOLD)
            all_indices = set(title_matches.keys()) | set(abs_matches.keys())
            
            if not all_indices:
                # Log failures with empty columns
                writer.writerow([q_id, query_texts[matrix_idx], "", "", "", "", "", "", 0.0, "No matches"])
                continue

            # --- Best Match Selection ---
            scored_results = []
            for idx in all_indices:
                s_t = title_matches.get(idx, -1.0)
                s_a = abs_matches.get(idx, -1.0)
                source = "title" if s_t > s_a else "abstract"
                scored_results.append((idx, max(s_t, s_a), source))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            best_idx, _, best_source = scored_results[0]
            best_row = df_corpus.iloc[int(best_idx)]
            
            # Prepare data for calculations
            ref_text_calc = str(best_row['title']) if best_source == 'title' else str(best_row['abstract'])
            best_title_clean = str(best_row.get('title', '')).strip().lower()

            # Prepare data for CSV (Both Title & Abstract)
            best_title_save = str(best_row.get('title', ''))
            best_abs_save = str(best_row.get('abstract', ''))

            # --- Deduplication ---
            candidate_indices = []
            candidate_texts = []
            for idx_tuple in scored_results[1:]:
                c_idx = int(idx_tuple[0])
                c_row = df_corpus.iloc[c_idx]
                if str(c_row.get('title', '')).strip().lower() == best_title_clean:
                    continue
                candidate_indices.append(c_idx)
                candidate_texts.append(str(c_row['title']) if best_source == 'title' else str(c_row['abstract']))
                if len(candidate_indices) >= run.TOP_K_CANDIDATES:
                    break

            if not candidate_indices:
                writer.writerow([q_id, query_texts[matrix_idx], best_idx, best_title_save, best_abs_save, "", "", "", 0.0, "No candidates"])
                continue

            # --- Reranking ---
            contra_idx, contra_score, _ = run.rerank_topk_and_find_contradiction(
                ref_text_calc, best_idx, best_source,
                candidate_indices, candidate_texts,
                retrieval_model, sparse_model, sparse_tokenizer,
                E_corpus, Es_corpus, run.ALPHA, run.DEVICE
            )

            if contra_idx is None:
                writer.writerow([q_id, query_texts[matrix_idx], best_idx, best_title_save, best_abs_save, "", "", "", 0.0, "No contra candidate"])
            else:
                contra_row = df_corpus.iloc[int(contra_idx)]
                
                # Prepare data for CSV (Both Title & Abstract)
                contra_title_save = str(contra_row.get('title', ''))
                contra_abs_save = str(contra_row.get('abstract', ''))
                
                # Write Full Row
                writer.writerow([
                    q_id, 
                    query_texts[matrix_idx], 
                    best_idx, 
                    best_title_save, 
                    best_abs_save,
                    contra_idx, 
                    contra_title_save,
                    contra_abs_save,
                    contra_score, 
                    "Ready for LLM"
                ])
                f.flush()

    print(f"\nStep 1 Complete. Data saved to {INTERMEDIATE_FILENAME}")

if __name__ == "__main__":
    main()