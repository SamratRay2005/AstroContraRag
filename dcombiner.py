import pandas as pd
import json
import glob
import os
import difflib

def get_similarity(s1, s2):
    """
    Returns a similarity score between 0.0 and 1.0.
    1.0 means strings are identical.
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        return 0.0
    # normalize strings (lower case, strip) to make comparison fair
    return difflib.SequenceMatcher(None, s1.strip().lower(), s2.strip().lower()).ratio()

def is_contradiction_too_similar(claim, contradictions_json, threshold=0.995):
    """
    Returns True if ANY contradiction in the list is >97% similar to the claim.
    """
    try:
        if pd.isna(contradictions_json) or pd.isna(claim):
            return True # Drop empty rows
            
        contradictions = json.loads(contradictions_json)
        
        for contra in contradictions:
            if not isinstance(contra, str):
                continue
            
            # CHECK: Calculate similarity
            score = get_similarity(claim, contra)
            
            # If similarity is > 97% (0.97), mark this row as bad
            if score > threshold:
                return True
            
            # Safety check: Remove very short garbage strings (< 5 chars)
            if len(contra.strip()) < 5:
                return True
                
        return False
        
    except (json.JSONDecodeError, TypeError, ValueError):
        return True # Drop unparseable rows

def main():
    files = sorted(glob.glob("train_claims_part_*.csv"))
    print(f"Found files: {files}")
    
    output_filename = "dataset.csv"
    total_rows = 0
    dropped_rows = 0
    kept_rows = 0
    
    chunk_list = []

    print(f"Processing files (Removing contradictions with >97% similarity to claim)...")
    
    for f_path in files:
        try:
            # 1. Read CSV, skipping corrupted lines
            df = pd.read_csv(f_path, on_bad_lines='skip', engine='python')
            
            current_len = len(df)
            total_rows += current_len
            
            # 2. Apply the >97% similarity filter
            # We want to KEEP rows where is_contradiction_too_similar is False
            mask = df.apply(lambda row: is_contradiction_too_similar(row.get('claim', ''), row.get('contradictions', '')), axis=1)
            df_clean = df[~mask]
            
            dropped_count = current_len - len(df_clean)
            dropped_rows += dropped_count
            kept_rows += len(df_clean)
            
            chunk_list.append(df_clean)
            print(f"  -> {f_path}: Scanned {current_len}, Filtered out {dropped_count} rows.")
            
        except Exception as e:
            print(f"  -> Critical error reading {f_path}: {e}")

    # 3. Save Result
    if chunk_list:
        final_df = pd.concat(chunk_list, ignore_index=True)
        final_df.to_csv(output_filename, index=False)
        
        print("\n" + "="*40)
        print(f"DONE! Saved to: {output_filename}")
        print(f"Total rows scanned:   {total_rows}")
        print(f"Rows dropped:  {dropped_rows}")
        print(f"Final valid dataset:  {kept_rows}")
        print("="*40)
    else:
        print("No data found to merge.")

if __name__ == "__main__":
    main()