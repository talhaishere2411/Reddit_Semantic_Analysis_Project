"""Fallback trim: 9.1 GB of JSONL -> Parquet with NO Spark, NO Java, NO Hadoop.

Use this if the winutils/hadoop.dll route fails. It produces the same
data/posts_parquet that d2_prep_trim.py would, so every later stage
(pipeline, verify, threshold, charts, report tables) works unchanged - those still
run on Spark, they just read Parquet instead of globbing JSONL.

    python d2_trim_pandas.py                # all of data/*.jsonl
    python d2_trim_pandas.py --batch 50000  # smaller memory chunks (default 100000)

How it avoids the two problems at once:
  * no glob through Hadoop  -> no NativeIO$Windows.access0 UnsatisfiedLinkError
  * streams line by line    -> never holds 9 GB in memory, only one batch

Output: data/posts_parquet/part-<n>.parquet  (one or more files per input file)
"""
import glob
import json
import os
import sys
import time

import pandas as pd

import d2_paths as P

BATCH = 100_000
for a in sys.argv[1:]:
    if a.startswith("--batch"):
        BATCH = int(a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1])

KEEP = ("id", "author", "subreddit", "created_utc", "score", "title", "selftext")
DEST = P.TRIM_DIR

files = P.list_jsonl()
if not files:
    raise SystemExit(f"[trim-py] no .jsonl in {P.DATA}/ — nothing to do")

os.makedirs(DEST, exist_ok=True)
for old in glob.glob(os.path.join(DEST, "*.parquet")):
    os.remove(old)                      # overwrite semantics, like mode('overwrite')

print("=" * 74)
print(f"PANDAS TRIM  ({len(files)} files, batch={BATCH:,})")
print("=" * 74)
print(f"  output -> {os.path.abspath(DEST)}")

total_read = total_kept = total_bad_json = 0
part = 0
t0 = time.time()

for fi, path in enumerate(files, 1):
    size_gb = os.path.getsize(path) / 1e9
    f_read = f_kept = f_bad = 0
    buf = []
    tf = time.time()

    def flush():
        global part, total_kept
        if not buf:
            return
        df = pd.DataFrame(buf, columns=list(KEEP))
        df = df.dropna(subset=["id", "created_utc"])
        df = df[df.id.astype(str).str.len() > 0]
        if len(df):
            df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
            df = df.dropna(subset=["created_utc"])
            df["ts"] = pd.to_datetime(df["created_utc"].astype("int64"), unit="s", utc=True)
            df["period"] = pd.cut(df["ts"],
                                  bins=[pd.Timestamp.min.tz_localize("UTC"),
                                        pd.Timestamp("2022-12-01", tz="UTC"),
                                        pd.Timestamp("2023-01-01", tz="UTC"),
                                        pd.Timestamp.max.tz_localize("UTC")],
                                  labels=["control_pre_LLM", "transition", "llm_era"],
                                  right=False)
            df["subreddit"] = df["subreddit"].astype("string").str.strip()
            for c in ("author", "title", "selftext"):
                df[c] = df[c].astype("string")
            df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype("int64")
            df = df.drop(columns=["created_utc"])
            out = os.path.join(DEST, f"part-{part:05d}.parquet")
            df.to_parquet(out, index=False, compression="snappy")
            part += 1
            total_kept += len(df)
        buf.clear()

    with open(path, encoding="utf8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            f_read += 1
            try:
                rec = json.loads(line)
            except Exception:
                f_bad += 1
                continue
            buf.append(tuple(rec.get(k) for k in KEEP))
            if len(buf) >= BATCH:
                flush()

    flush()
    total_read += f_read
    total_bad_json += f_bad
    el = round(time.time() - tf, 1)
    rate = f_read / el if el else 0
    print(f"  [{fi:2}/{len(files)}] {os.path.basename(path):<38} {size_gb:5.2f} GB  "
          f"read={f_read:>8,}  bad_json={f_bad:>6,}  {el:>6.1f}s  {rate:>9,.0f} rec/s",
          flush=True)

el = round(time.time() - t0, 1)
print("\n" + "=" * 74)
print("RESULT")
print("=" * 74)
print(f"  rows read from jsonl : {total_read:,}")
print(f"  rows kept            : {total_kept:,}")
print(f"  unparseable lines    : {total_bad_json:,}")
print(f"  dropped (null id/ts) : {total_read - total_bad_json - total_kept:,}")
print(f"  parquet files        : {part}")
print(f"  elapsed              : {el} s")
if total_kept == 0:
    raise SystemExit("[trim-py] STOP: nothing written. Check that created_utc is numeric.")
if total_kept < total_read * 0.5:
    print(f"  !! only {100*total_kept/max(total_read,1):.1f}% kept — inspect before trusting")

# ---- report cells from the result, read back with pandas (not Spark) ----
print("\nLoading the Parquet back to compute the report cells ...", flush=True)
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(DEST, "*.parquet")))],
               ignore_index=True)
print(f"\n  TOTAL POSTS: {len(df):,}")
print("\n=== Table 6.5 row counts (period) ===")
print(df["period"].value_counts().to_string())
print("\n=== Table 6.6: subreddit x period ===")
print(pd.crosstab(df["subreddit"], df["period"]).to_string())
print("\n=== observed CONTROL range -> decides the Table 6.5 row LABEL ===")
c = df[df.period == "control_pre_LLM"]
print(f"  first month : {c['ts'].min()}")
print(f"  last month  : {c['ts'].max()}")
print(f"  rows        : {len(c):,}")
e = df[df.period == "llm_era"]
print(f"  llm_era     : {e['ts'].min()} .. {e['ts'].max()}   rows {len(e):,}")
print(f"\n  distinct subreddits: {df['subreddit'].nunique()}  -> {sorted(df['subreddit'].dropna().unique())}")
print("""
NEXT
  1. the pipeline still runs on Spark - it reads data/posts_parquet, no globbing
  2. Table 6.1 fraction = 1.0
  3. note in section 7 that the one-off JSONL->Parquet conversion ran in pandas
""")
