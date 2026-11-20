"""Aggregation-correctness check for Dataset 2 (Table 7.1, row 2).

Spark groupBy counts vs pandas value_counts on a 100,000-row sample; expects MATCH.
Standalone version of verify_aggregation.py that points at the Dataset 2 output instead
of output/labelled.parquet, so you do not have to edit your own script.

    python d2_verify_aggregation.py [posts]
"""
import os
import sys

import d2_paths as P

KIND = sys.argv[1] if len(sys.argv) > 1 else "posts"
PATH = os.path.join(P.OUT_D2, f"{KIND}_scored.parquet")

if not os.path.isdir(PATH):
    raise SystemExit(f"[verify] missing {PATH} — run ai_comments_pipeline.py first")

spark = P.spark("d2_verify")
sample = spark.read.parquet(PATH).limit(100_000).cache()

spark_counts = {r["label"]: r["n"] for r in
                sample.groupBy("label").count().withColumnRenamed("count", "n").collect()}
pdf = sample.select("label").toPandas()
pandas_counts = {str(k): int(v) for k, v in pdf["label"].value_counts().to_dict().items()}

print("Spark :", spark_counts)
print("pandas:", pandas_counts)
print("rows checked:", f"{len(pdf):,}")
print("MATCH" if spark_counts == pandas_counts else "MISMATCH")
print("\nPut this line in §7.1 for Dataset 2, e.g.:")
print(f'  "Spark counts: {spark_counts}. pandas counts: {pandas_counts}. Result: '
      f'{"MATCH" if spark_counts == pandas_counts else "MISMATCH"} ({len(pdf):,} rows)."')
spark.stop()
