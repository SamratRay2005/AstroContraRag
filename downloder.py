from datasets import load_dataset
import csv, json, os, time, traceback
from datetime import date, datetime

dataset_name = "kiyer/pathfinder_arxiv_data"
out_file = "pathfinder_arxiv_train.csv"
checkpoint_file = out_file + ".ckpt"
bad_rows_file = out_file + ".bad"
buffer_size = 1000
max_retries = 3
resume = True 

def _serialize_value(v):
    # FIX: Handle date and datetime objects by converting them to string
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    
    # Fallback for lists/dicts
    return json.dumps(v, ensure_ascii=False)

# read checkpoint
rows_already_written = 0
if resume and os.path.exists(checkpoint_file):
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as ck:
            content = ck.read().strip()
            if content:
                rows_already_written = int(content)
    except Exception:
        rows_already_written = 0

mode = "a" if resume and os.path.exists(out_file) else "w"
print(f"Starting download. Resuming from row {rows_already_written}...")

ds = load_dataset(dataset_name, split="train", streaming=True)

# OPTIMIZATION: Skip rows natively instead of looping in Python
if rows_already_written > 0:
    print(f"Fast-forwarding dataset stream by {rows_already_written} rows... (Please wait)")
    ds = ds.skip(rows_already_written)

with open(out_file, mode, newline="", encoding="utf-8") as f_out, \
     open(bad_rows_file, "a", encoding="utf-8") as badf:

    writer = None
    buf = []
    
    try:
        # Note: 'i' resets to 0 after skip(), so we calculate actual_index manually
        for i, row in enumerate(ds):
            actual_index = rows_already_written + i 

            # Lazy writer init
            if writer is None:
                headers = list(row.keys())
                # FIX: extrasaction='ignore' prevents crashes if new rows have extra columns
                writer = csv.DictWriter(f_out, fieldnames=headers, extrasaction='ignore')
                if mode == "w":
                    writer.writeheader()
                    f_out.flush()

            # Serialize and buffer
            success = False
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    serialized = {k: _serialize_value(v) for k, v in row.items()}
                    buf.append(serialized)
                    success = True
                    break
                except Exception as e:
                    last_exc = e
                    wait = 0.5 * (2 ** (attempt - 1))
                    # Only print if it's NOT a keyboard interrupt
                    print(f"\nRow {actual_index} error (attempt {attempt}): {e}")
                    time.sleep(wait)

            if not success:
                badf.write(f"ROW_INDEX:{actual_index}\tERROR:{str(last_exc)}\n")
                badf.flush()
                continue

            # Flush to disk
            if len(buf) >= buffer_size:
                writer.writerows(buf)
                f_out.flush()
                
                # Update checkpoint counter by the buffer size
                # Note: We don't use 'i' here because we want the total count
                current_total = rows_already_written + len(buf)
                
                # Clear buffer first, then update tracker
                buf.clear()
                
                # Update global tracker
                rows_already_written = current_total

                # Update checkpoint file
                with open(checkpoint_file, "w", encoding="utf-8") as ck:
                    ck.write(str(rows_already_written))
                
                print(f"Saved {rows_already_written} rows...", end="\r")

        # Flush remaining
        if buf:
            writer.writerows(buf)
            f_out.flush()
            rows_already_written += len(buf)
            with open(checkpoint_file, "w", encoding="utf-8") as ck:
                ck.write(str(rows_already_written))

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving buffer...")
        if writer and buf:
            writer.writerows(buf)
            f_out.flush()
            rows_already_written += len(buf)
            with open(checkpoint_file, "w", encoding="utf-8") as ck:
                ck.write(str(rows_already_written))
        print(f"Progress saved at row {rows_already_written}. You can resume later.")
        raise

print(f"\nDone! Total rows written: {rows_already_written}")