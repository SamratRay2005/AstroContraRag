import pandas as pd
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# Initialize Groq client
# Ensure your GROQ_API_KEY is set in your environment variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def evaluate_contradiction():
    input_file = 'intermediate_candidates.csv'
    
    # Load the data
    df = pd.read_csv(input_file)
    
    # Initialize the column if it doesn't exist
    if 'llm_verified' not in df.columns:
        df['llm_verified'] = ""

    print(f"Starting evaluation of {len(df)} rows...")

    y_count = 0

    for index, row in df.iterrows():
        # Skip rows already processed in case of a restart
        if pd.notna(row['llm_verified']) and row['llm_verified'] != "":
            if row['llm_verified'] == 'Y': y_count += 1
            continue

        prompt = f"""Analyze these two scientific abstracts.
        
        ABSTRACT A (Primary): {row['best_match_abstract']}
        
        ABSTRACT B (Comparison): {row['contra_match_abstract']}
        
        TASK:
        Does Abstract B represent a logical contradiction, an opposing result, or a different viewpoint on the SAME specific phenomenon/research question as Abstract A?
        
        - Answer 'Y' if they are about the same topic but offer conflicting or different evidence/POVs.
        - Answer 'N' if they are simply about different topics, different objects, or are unrelated research.
        
        Answer ONLY with 'Y' or 'N'."""

        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # Low temperature for consistency
                max_tokens=2,
                top_p=1,
                stream=False
            )

            result = completion.choices[0].message.content.strip().upper()
            
            # Clean response to ensure it's just Y or N
            final_val = 'Y' if 'Y' in result else 'N'
            
            # Update DataFrame and save immediately
            df.at[index, 'llm_verified'] = final_val
            df.to_csv(input_file, index=False)
            
            if final_val == 'Y':
                y_count += 1
            
            print(f"Row {index}: LLM Result = {final_val}")

        except Exception as e:
            print(f"Error on row {index}: {e}")
            continue

    # Final Accuracy Calculation
    total_processed = len(df[df['llm_verified'] != ""])
    accuracy = (y_count / total_processed) * 100 if total_processed > 0 else 0
    
    print("\n" + "="*30)
    print(f"Evaluation Complete!")
    print(f"Total Valid Contradictions (Y): {y_count}")
    print(f"Total Rows Processed: {total_processed}")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print("="*30)

if __name__ == "__main__":
    evaluate_contradiction()