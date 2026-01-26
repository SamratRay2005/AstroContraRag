### K = 200:-

```console
Terminal:- python ht.py
Loading resources...
Loading metadata from arxiv_metadata.parquet...
Loading dataset: train.csv.gz...
Loaded 10000 rows. Sampling 500 for mixed testing...

--- Simulating User Queries ---
Dataset prepared: 235 fuzzy queries / 265 exact queries.
Example Fuzzy Query: 'for IACTs on indirect searches'

--- Encoding Queries ---
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:02<00:00,  7.49it/s]

--- Running Initial Search (Top 200) ---

--- Testing 784 Combinations ---
Tested 100 combos...
Tested 200 combos...
Tested 300 combos...
Tested 400 combos...
Tested 500 combos...
Tested 600 combos...
Tested 700 combos...

--- Tuning Results (Top 10 by Recall, then Efficiency) ---
 Title_Thresh  Abs_Thresh  Recall (%)  Avg_Results
      0.57         0.71         98.8       258.686
      0.55         0.71         98.8       260.808
      0.53         0.71         98.8       261.388
      0.51         0.71         98.8       261.810
      0.49         0.71         98.8       262.230
      0.47         0.71         98.8       262.378
      0.35         0.71         98.8       262.682
      0.37         0.71         98.8       262.682
      0.39         0.71         98.8       262.682
      0.41         0.71         98.8       262.682

>>> SUGGESTED SETTINGS <<<
TITLE_THRESHOLD = 0.57
ABSTRACT_THRESHOLD = 0.71
Stats: 98.80% Recall with approx 258.7 results/query

Saved full results to 'hyperparameter_mixed_results.csv'

```

---

### Currently in use:-

#### K = 20:-

```console
:- python ht.py
Loading resources...
Loading metadata from arxiv_metadata.parquet...
Loading dataset: train.csv.gz...
Loaded 10000 rows. Sampling 500 for mixed testing...

--- Simulating User Queries ---
Dataset prepared: 245 fuzzy queries / 255 exact queries.
Example Fuzzy Query: 'Pointing optimization for IACTs on indirect dark matter searches'

--- Encoding Queries ---
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:02<00:00,  6.16it/s]

--- Running Initial Search (Top 20) ---

--- Testing 784 Combinations ---
Tested 100 combos...
Tested 200 combos...
Tested 300 combos...
Tested 400 combos...
Tested 500 combos...
Tested 600 combos...
Tested 700 combos...

--- Tuning Results (Top 10 by Recall, then Efficiency) ---
 Title_Thresh  Abs_Thresh  Recall (%)  Avg_Results
      0.53         0.87         96.6        19.962
      0.53         0.89         96.6        19.962
      0.55         0.87         96.6        19.962
      0.55         0.89         96.6        19.962
      0.51         0.87         96.6        19.968
      0.51         0.89         96.6        19.968
      0.53         0.85         96.6        19.976
      0.55         0.85         96.6        19.976
      0.49         0.87         96.6        19.982
      0.49         0.89         96.6        19.982

>>> SUGGESTED SETTINGS <<<
TITLE_THRESHOLD = 0.53
ABSTRACT_THRESHOLD = 0.87
Stats: 96.60% Recall with approx 20.0 results/query

Saved full results to 'hyperparameter_mixed_results.csv'

```