"""Export a balanced hand-labelling sample for the AI-likeness validation (Table 7.4).

Replaces the 'export' half of validate_sample.py, which loads every jsonl line into
pandas and would exhaust memory on the real data.

    python d2_validate_export_spark.py [n]        # default 200 -> 200 flagged + 200 unflagged

Reads the raw text from the trimmed Parquet when it exists (much cheaper than
re-parsing the jsonl), joins it to the scored Parquet, and writes
output_ai/validation_sample.csv with a blank human_label column.

Then: label every row AI or HUMAN, and run  python validate_sample.py score

The sample is balanced by construction, so precision-of-flagged and
share-of-unflagged-that-is-AI are valid; recall is NOT - do not report it.
The CSV contains raw user text: keep it out of the submitted repo.
"""
import csv
import sys

import d2_paths as P
from pyspark.sql import functions as F

KIND = "posts"
N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200
SEED = 42

P.preflight()
P.ensure_dirs(P.OUT_D2)
spark = P.spark("d2_validate_export")

scored_path = f"{P.OUT_D2}/{KIND}_scored.parquet"
scored = (spark.read.parquet(scored_path)
               .filter(F.col("period") == "llm_era")
               .select("id", "ai_like", "ai_score"))

if P.have_trim():
    raw = spark.read.parquet(P.spark_dir()).select(
        "id", F.concat_ws(" ", F.col("title"),
                          F.coalesce(F.col("selftext"), F.lit(""))).alias("text"))
else:
    raw = spark.read.json(P.spark_glob()).select(
        "id", F.concat_ws(" ", F.col("title"),
                          F.coalesce(F.col("selftext"), F.lit(""))).alias("text"))

m = scored.join(raw, on="id", how="inner").cache()
n_pos = m.filter(F.col("ai_like") == 1).count()
n_neg = m.filter(F.col("ai_like") == 0).count()
print(f"[validate] llm_era flagged={n_pos:,}  unflagged={n_neg:,}")
if min(n_pos, n_neg) < N:
    print(f"[validate] WARNING: fewer than {N} rows in one group; sample will be unbalanced.")

pos = m.filter(F.col("ai_like") == 1).orderBy(F.rand(SEED)).limit(N)
neg = m.filter(F.col("ai_like") == 0).orderBy(F.rand(SEED)).limit(N)
out = pos.unionByName(neg).orderBy(F.rand(1)).toPandas()      # ~400 rows: safe
out["human_label"] = ""

dest = f"{P.OUT_D2}/validation_sample.csv"
out[["id", "text", "ai_like", "ai_score", "human_label"]].to_csv(
    dest, index=False, quoting=csv.QUOTE_ALL)
print(f"[validate] wrote {len(out)} rows to {dest}")
print("[validate] label EVERY row AI or HUMAN (sort by id first and hide ai_like so you")
print("           are not anchored), then:  python validate_sample.py score")
spark.stop()
