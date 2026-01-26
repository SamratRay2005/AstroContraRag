import pandas as pd
import gzip
import random
from io import StringIO

DATA_FILE_PATH = "train.csv.gz"
SAMPLE_SIZE = 4500

print("Starting single-pass random sampling...")

# Reservoir Sampling:
# We process the file line-by-line as raw strings (fast) 
# and only parse the CSV data at the very end.

with gzip.open(DATA_FILE_PATH, 'rt') as f:
    # 1. Grab the header first
    header = next(f)
    
    # 2. Fill the "reservoir" with the first 4500 lines
    reservoir = []
    for i, line in enumerate(f):
        if i < SAMPLE_SIZE:
            reservoir.append(line)
        else:
            # 3. For every subsequent line, effectively flip a coin to see if we keep it.
            # The probability of keeping the 'i-th' line is (SAMPLE_SIZE / i).
            # This mathematically guarantees a uniform random sample over the whole file.
            j = random.randint(0, i)
            if j < SAMPLE_SIZE:
                reservoir[j] = line

print("Sampling complete. Parsing selected data...")

# 4. Combine header and selected lines into a single string
final_data_str = header + "".join(reservoir)

# 5. Parse only this small subset into Pandas
df_final = pd.read_csv(StringIO(final_data_str))

# 6. Save
output_filename = "finaltest.csv"
df_final.to_csv(output_filename, index=False)

print(f"Success! Saved {len(df_final)} random rows to '{output_filename}'.")