import pandas as pd
import os
import time
import sys
from groq import Groq, RateLimitError, APIStatusError
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION: LIST OF API KEYS ---
API_KEYS = [
    "api_1", # Primary
    "api_2", # Backup 1
    "api_3", # Backup 2
]

class KeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.current_index = 0
        self.client = Groq(api_key=self.keys[0])
        print(f"Initialized with Key #{self.current_index + 1}")

    def switch_key(self):
        """Switches to the next available API key."""
        self.current_index = (self.current_index + 1) % len(self.keys)
        new_key = self.keys[self.current_index]
        self.client = Groq(api_key=new_key)
        print(f"\n!!! SWITCHING API KEY !!! Now using Key #{self.current_index + 1}\n")
        return self.current_index

# Initialize the manager
key_manager = KeyManager(API_KEYS)

def evaluate_contradiction():
    input_file = 'intermediate_candidates.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    
    # 1. Setup Column
    if 'llm_verified' not in df.columns:
        df['llm_verified'] = ""
    
    # Clean up any NaN/None to ensure strict string checking
    df['llm_verified'] = df['llm_verified'].fillna("").astype(str).replace('nan', '')

    # 2. FIND THE POINTER (The Start Index)
    empty_rows = df[df['llm_verified'] == ""]
    
    if empty_rows.empty:
        print("All rows have already been processed!")
        return

    start_index = empty_rows.index[0]
    total_rows = len(df)
    
    # Calculate stats for the user
    y_count = df.iloc[:start_index]['llm_verified'].apply(lambda x: 1 if 'Y' in x.upper() else 0).sum()

    print("="*30)
    print(f"Total Rows      : {total_rows}")
    print(f"Pointer Resuming: Row {start_index}")
    print(f"Previous Y Count: {y_count}")
    print("="*30)

    batch_counter = 0

    # 3. LOOP (Starts exactly at pointer)
    for index, row in empty_rows.iterrows():
        
        prompt = f"""Analyze these two scientific abstracts.
        
        ABSTRACT A (Primary): {row['best_match_abstract']}
        
        ABSTRACT B (Comparison): {row['contra_match_abstract']}
        
        TASK:
        Does Abstract B represent a logical contradiction, an opposing result, or a different viewpoint on the SAME specific phenomenon/research question as Abstract A?
        
        - Answer 'Y' if they are about the same topic but offer conflicting or different evidence/POVs.
        - Answer 'N' if they are simply about different topics, different objects, or are unrelated research.
        
        Answer ONLY with 'Y' or 'N'."""

        success = False
        keys_tried = 0 
        
        # RETRY LOOP
        while not success:
            # CHECK: Have we exhausted all keys for this row?
            if keys_tried >= len(API_KEYS):
                print(f"\n[CRITICAL] All {len(API_KEYS)} API keys are exhausted/rate-limited.")
                
                # --- NEW SECTION: Calculate & Show Accuracy Before Stopping ---
                processed_mask = df['llm_verified'] != ""
                total_proc = processed_mask.sum()
                final_y_count = df.loc[processed_mask, 'llm_verified'].apply(lambda x: 1 if 'Y' in str(x).upper() else 0).sum()
                final_acc = (final_y_count / total_proc) * 100 if total_proc > 0 else 0
                
                print("\n" + "="*30)
                print(f"STATUS REPORT (Stopped at Row {index})")
                print(f"Total Rows Processed: {total_proc}")
                print(f"Total Valid (Y)     : {final_y_count}")
                print(f"Current Accuracy    : {final_acc:.2f}%")
                print("="*30)
                # -------------------------------------------------------------

                print("Saving progress and stopping script safely.")
                df.to_csv(input_file, index=False)
                sys.exit(0) 

            try:
                # Use the client from the key_manager
                completion = key_manager.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2,
                    stream=False
                )

                result = completion.choices[0].message.content.strip().upper()
                final_val = 'Y' if 'Y' in result else 'N'
                
                df.at[index, 'llm_verified'] = final_val
                if final_val == 'Y': y_count += 1
                
                print(f"Row {index} Processed: {final_val} (Total Ys: {y_count})")
                success = True 

            except (RateLimitError, APIStatusError) as e:
                print(f"API Error on Key #{key_manager.current_index + 1}: {e}")
                key_manager.switch_key()
                keys_tried += 1
                time.sleep(1) 
                
            except Exception as e:
                print(f"Critical Unknown Error on row {index}: {e}")
                break 

        # Batch Save Logic
        batch_counter += 1
        if batch_counter >= 15:
            df.to_csv(input_file, index=False)
            print(f"   >>> Checkpoint saved at row {index}")
            batch_counter = 0

    # Final Save
    df.to_csv(input_file, index=False)
    
    # Final Success Report
    total_proc_end = len(df[df['llm_verified'] != ""])
    final_acc_end = (y_count / total_proc_end) * 100 if total_proc_end > 0 else 0
    print("\n" + "="*30)
    print("Evaluation Complete!")
    print(f"Final Accuracy: {final_acc_end:.2f}%")
    print(">>> Final save complete.")

if __name__ == "__main__":
    evaluate_contradiction()