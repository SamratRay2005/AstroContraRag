import os
import time
import csv
import re
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.multiprocessing as mp

# ---------- CONFIG ----------
HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
IN_CSV = "train.csv.gz"
# Output filename pattern: will become train_claims_part_0.csv, part_1.csv, etc.
OUT_CSV_TEMPLATE = "train_claims_part_{rank}.csv"

# Generation Params
NUM_A_PER_CLAIM = 4
NUM_B_PER_CLAIM = 3
MAX_CLAIMS_CANDIDATES = 12
MAX_CLAIMS_TO_SELECT = 5

TOKEN_OVERLAP_THRESHOLD_CLAIM = 0.40
TOKEN_OVERLAP_THRESHOLD_ABSTRACT = 0.15

# Output fields
FIELDNAMES = [
    "orig_row_index","arxiv_id","title","authors","abstract",
    "selected_claim_rank","claim_idx_in_candidates","claim","paraphrase",
    "contradictions","contradiction_types","variant_indices","strategies","statuses",
    "status_overall","timestamp"
]

# ==========================================
# ROBUST PROMPTS (MODIFIED)
# ==========================================

EXTRACT_CLAIMS_PROMPT = """
Instruction: Analyze the abstract below and extract {max_claims} Atomic Factual Claims.
Criteria for "Atomic":
1. Self-contained: Replace pronouns (it, they, this method) with the specific noun entities they refer to.
2. Distinct: Do not combine two separate findings into one sentence.
3. Explicit: Must be directly supported by the text.

Abstract:
{abstract}

Output ONLY the numbered list of claims:
1.
""".strip()

PARAPHRASE_PROMPT = """
Instruction: Rewrite the following claim to be semantically identical but syntactically distinct. 
- Change the grammatical structure (e.g., Active <-> Passive).
- Use synonyms where possible.
- Do NOT add or remove information.

Claim:
{claim}

Paraphrase:
""".strip()

# TYPE A: Distractors (Same vocabulary, broken logic)
# We want sentences that 'look' like the claim to a keyword searcher, but mean nothing.
CONTRAD_A_PROMPT_CLAIM_ONLY = """
Instruction: Rearrange the words and entities in the CLAIM below to create a "Hallucination" that sounds plausible but is logically incoherent or unrelated to the original fact.
- Constraint: You must use mostly the same words as the original claim.
- Goal: Create a "distractor" sentence that has high lexical overlap but false meaning.

Claim:
{claim}

Strategy: {strategy}

HallucinatedStatement:
""".strip()

# TYPE B: Hard Negatives (Subtle Logical Inversions)
# We want to force the model to look for relations, not just keywords.
CONTRAD_B_PROMPT_CLAIM_ONLY = """
Instruction: Generate a "Hard Negative" contradiction for the CLAIM. 
- Constraint: Do NOT simply add the word "not" or "no". 
- Method: You must swap entities, invert a causal relationship, or change a specific metric/number to make the statement false.
- The result must be a plausible scientific sentence, just factually wrong.

Claim:
{claim}

Strategy: {strategy}

HardNegative:
""".strip()

TYPE_A_STRATEGIES = [
    "Swap the subject and object to create a category error.",
    "Combine fragments from the start and end of the claim to create a nonsensical statement.",
    "Keep the entities but assign them an unrelated action/predicate from the claim.",
    "Scramble the adjectives to modify the wrong nouns."
]

TYPE_B_STRATEGIES = [
    "Invert the causal direction (e.g., 'A causes B' -> 'B causes A').",
    "Replace the specific algorithm/method name with a competing one mentioned or implied.",
    "Change the numerical result (e.g., 'significant increase' -> 'negligible change').",
    "Swap the dependent and independent variables."
]

# ==========================================
# END PROMPTS
# ==========================================

# ---------- HELPER FUNCTIONS (Stateless) ----------

def parse_numbered_list(text):
    items = []
    for line in text.splitlines():
        m = re.match(r'^\s*\d+\.\s*(.+)$', line)
        if m:
            items.append(m.group(1).strip().rstrip(' .'))
    if not items:
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"])', text)
        items = [s.strip().rstrip('.') for s in sents if s.strip()]
    return items

def split_sentences_heuristic(text):
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"])', (text or "").strip())
    sents = [s.strip().rstrip('.') for s in sents if s.strip()]
    return sents

def tokens_set(text):
    return set(re.findall(r"[A-Za-z0-9\-\']+", (text or "").lower()))

def token_overlap_ratio(candidate, reference):
    cw = set(re.findall(r"[A-Za-z0-9\-\']+", (candidate or "").lower()))
    rw = set(re.findall(r"[A-Za-z0-9\-\']+", (reference or "").lower()))
    if not cw:
        return 0.0
    return len(cw & rw) / len(cw)

def greedy_claim_selection(candidate_claims, abstract_tokens, k):
    cand_tokens = [(idx, tokens_set(claim)) for idx, claim in candidate_claims]
    selected = []
    covered = set()
    remaining = cand_tokens.copy()
    for _ in range(k):
        best = None
        best_new = -1
        for idx, toks in remaining:
            new_count = len((toks & abstract_tokens) - covered)
            if new_count > best_new:
                best_new = new_count
                best = (idx, toks)
        if best is None or best_new == 0:
            break
        selected_idx = best[0]
        claim_text = next(c for (i, c) in candidate_claims if i == selected_idx)
        selected.append((selected_idx, claim_text))
        covered |= (best[1] & abstract_tokens)
        remaining = [t for t in remaining if t[0] != selected_idx]
    
    if len(selected) < k:
        remaining_claims = [ (i, c) for (i,c) in candidate_claims if (i,c) not in selected ]
        rem_sorted = sorted(remaining_claims, key=lambda ic: len(tokens_set(ic[1]) & abstract_tokens), reverse=True)
        for i,c in rem_sorted:
            if len(selected) >= k: break
            if (i,c) not in selected: selected.append((i,c))
    return selected[:k]

def append_row_atomic(path, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        try: os.fsync(f.fileno())
        except OSError: pass

# ---------- MODEL WORKER ----------

def call_local_model(model, tokenizer, prompt, temp=0.7, top_p=0.9, max_new_tokens=256):
    try:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()
    except Exception as e:
        print(f"[WARN] Inference error: {e}")
        return ""

def process_shard(rank, df_shard):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    print(f"[Rank {rank}] Start. PID: {os.getpid()} on GPU {rank}. Rows: {len(df_shard)}")

    my_out_csv = OUT_CSV_TEMPLATE.format(rank=rank)

    print(f"[Rank {rank}] Loading Model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"[Rank {rank}] CRITICAL ERROR loading model: {e}")
        return

    existing = set()
    if os.path.exists(my_out_csv):
        try:
            done = pd.read_csv(my_out_csv, usecols=["orig_row_index","selected_claim_rank"])
            for _, r in done.iterrows():
                existing.add((int(r["orig_row_index"]), int(r["selected_claim_rank"])))
            print(f"[Rank {rank}] Resumed {len(existing)} rows.")
        except Exception:
            pass

    for _, row in df_shard.iterrows():
        orig_idx = row["orig_idx_preserves"] 
        arxiv_id = str(row.get("arxiv_id",""))
        title = str(row.get("title",""))
        abstract = str(row.get("abstract","")).strip()
        authors = str(row.get("authors",""))

        if not abstract:
            continue

        candidate_claims = []
        try:
            # Lowered temp for extraction to be more precise
            prompt = EXTRACT_CLAIMS_PROMPT.format(abstract=abstract, max_claims=MAX_CLAIMS_CANDIDATES)
            resp = call_local_model(model, tokenizer, prompt, temp=0.1, top_p=0.9, max_new_tokens=600)
            candidate_claims = parse_numbered_list(resp)[:MAX_CLAIMS_CANDIDATES]
        except Exception as e:
            print(f"[Rank {rank}] Claim extraction error {orig_idx}: {e}")

        if not candidate_claims:
            sents = split_sentences_heuristic(abstract)
            candidate_claims = sents[:MAX_CLAIMS_CANDIDATES]

        cleaned = []
        seen = set()
        for c in candidate_claims:
            cshort = c.strip()
            if len(cshort) < 8: continue
            lc = cshort.lower()
            if lc in seen: continue
            seen.add(lc)
            cleaned.append(cshort)
        candidate_claims = cleaned

        if not candidate_claims:
            continue

        candidate_list_indexed = list(enumerate(candidate_claims))
        abstract_tokens = tokens_set(abstract)
        selected_claims = greedy_claim_selection(candidate_list_indexed, abstract_tokens, MAX_CLAIMS_TO_SELECT)

        if not selected_claims:
            sorted_by_cov = sorted(candidate_list_indexed, key=lambda ic: len(tokens_set(ic[1]) & abstract_tokens), reverse=True)
            selected_claims = sorted_by_cov[:MAX_CLAIMS_TO_SELECT]

        for rank_in_row, (claim_idx_in_candidates, claim_text) in enumerate(selected_claims):
            if (int(orig_idx), int(rank_in_row)) in existing:
                continue

            paraphrase = ""
            try:
                # Higher temp for paraphrase diversity
                para_prompt = PARAPHRASE_PROMPT.format(claim=claim_text)
                paraphrase = call_local_model(model, tokenizer, para_prompt, temp=0.4, max_new_tokens=150)
                paraphrase = paraphrase.splitlines()[0].strip()
            except Exception: pass

            W_CLAIM = 0.7
            W_ABS = 0.3
            a_candidates = []
            b_candidates = []

            for v in range(NUM_A_PER_CLAIM):
                strategy = TYPE_A_STRATEGIES[v % len(TYPE_A_STRATEGIES)]
                prompt = CONTRAD_A_PROMPT_CLAIM_ONLY.format(claim=claim_text, strategy=strategy)
                
                # Higher temp for hallucinations
                produced = call_local_model(model, tokenizer, prompt, temp=0.8, top_p=0.95, max_new_tokens=220)
                produced = produced.splitlines()[0].strip() if produced else ""
                
                status = "ok"
                ov_claim = token_overlap_ratio(produced, claim_text)
                ov_abs = token_overlap_ratio(produced, abstract)
                if ov_claim < TOKEN_OVERLAP_THRESHOLD_CLAIM or ov_abs < TOKEN_OVERLAP_THRESHOLD_ABSTRACT:
                    status = "invalid"
                if not produced: status = "error"

                a_candidates.append({
                    "produced": produced, "status": status, "variant_idx": v,
                    "strategy": strategy, "score": W_CLAIM * ov_claim + W_ABS * ov_abs
                })

            for v in range(NUM_B_PER_CLAIM):
                idx_v = NUM_A_PER_CLAIM + v
                strategy = TYPE_B_STRATEGIES[v % len(TYPE_B_STRATEGIES)]
                prompt = CONTRAD_B_PROMPT_CLAIM_ONLY.format(claim=claim_text, strategy=strategy)
                
                # Moderate temp for hard negatives
                produced = call_local_model(model, tokenizer, prompt, temp=0.6, top_p=0.85, max_new_tokens=180)
                produced = produced.splitlines()[0].strip() if produced else ""

                status = "ok"
                ov_claim = token_overlap_ratio(produced, claim_text)
                ov_abs = token_overlap_ratio(produced, abstract)
                if ov_claim < TOKEN_OVERLAP_THRESHOLD_CLAIM or ov_abs < TOKEN_OVERLAP_THRESHOLD_ABSTRACT:
                    status = "invalid"
                if not produced: status = "error"

                b_candidates.append({
                    "produced": produced, "status": status, "variant_idx": idx_v,
                    "strategy": strategy, "score": W_CLAIM * ov_claim + W_ABS * ov_abs
                })

            def pick_best(cands):
                ok = [c for c in cands if c["status"] == "ok" and c["produced"]]
                if ok: return max(ok, key=lambda x: x["score"])
                prod = [c for c in cands if c["produced"]]
                if prod: return max(prod, key=lambda x: x["score"])
                return None

            best_a = pick_best(a_candidates)
            best_b = pick_best(b_candidates)

            contradictions = []
            contradiction_types = []
            variant_indices = []
            strategies = []
            statuses = []

            if best_a:
                contradictions.append(best_a["produced"])
                contradiction_types.append("A")
                variant_indices.append(best_a["variant_idx"])
                strategies.append(best_a["strategy"])
                statuses.append(best_a["status"])
            else:
                contradictions.append("")
                contradiction_types.append("A")
                variant_indices.append(None)
                strategies.append(None)
                statuses.append("missing")

            if best_b:
                contradictions.append(best_b["produced"])
                contradiction_types.append("B")
                variant_indices.append(best_b["variant_idx"])
                strategies.append(best_b["strategy"])
                statuses.append(best_b["status"])
            else:
                contradictions.append("")
                contradiction_types.append("B")
                variant_indices.append(None)
                strategies.append(None)
                statuses.append("missing")

            ok_count = sum(1 for s in statuses if s == "ok")
            if ok_count == 2: status_overall = "ok"
            elif ok_count == 1: status_overall = "partial"
            elif any(s in ("invalid","error") for s in statuses): status_overall = "partial"
            else: status_overall = "invalid"

            out_row = {
                "orig_row_index": orig_idx,
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "selected_claim_rank": rank_in_row,
                "claim_idx_in_candidates": claim_idx_in_candidates,
                "claim": claim_text,
                "paraphrase": paraphrase,
                "contradictions": json.dumps(contradictions, ensure_ascii=False),
                "contradiction_types": json.dumps(contradiction_types, ensure_ascii=False),
                "variant_indices": json.dumps(variant_indices, ensure_ascii=False),
                "strategies": json.dumps(strategies, ensure_ascii=False),
                "statuses": json.dumps(statuses, ensure_ascii=False),
                "status_overall": status_overall,
                "timestamp": datetime.utcnow().isoformat()
            }

            append_row_atomic(my_out_csv, out_row)
            print(f"[Rank {rank}] Completed orig={orig_idx} claim={rank_in_row} status={status_overall}")

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("[MAIN] Reading input CSV...")
    df = pd.read_csv(IN_CSV, low_memory=False).fillna("")
    df['orig_idx_preserves'] = df.index
    
    NUM_GPUS = 4
    chunks = np.array_split(df, NUM_GPUS)
    processes = []

    print(f"[MAIN] Spawning {NUM_GPUS} processes...")
    
    for rank in range(NUM_GPUS):
        p = mp.Process(target=process_shard, args=(rank, chunks[rank]))
        p.start()
        processes.append(p)
        print(f"[MAIN] Process {rank} started.")

    for p in processes:
        p.join()

    print("[MAIN] All processes completed.")

if __name__ == "__main__":
    main()