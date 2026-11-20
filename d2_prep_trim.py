"""One-off trim: the ~10 GB of JSONL in data/ -> ~1.2 GB Parquet in data/posts_parquet/.

Run it once. Everything downstream then reads the Parquet in seconds and the raw
jsonl can be deleted.

    python d2_prep_trim.py                 # auto-detects data/*.jsonl
    python d2_prep_trim.py --master "local[4]"

Why: ai_comments_pipeline.py does spark.read.json(PATH) with no schema, so Spark scans
every byte to infer a schema for ~80 fields and then the next line discards 73 of them.
That inference pass repeats on every run. Here the schema is explicit, so it does not
happen at all, and the fields that survive are only the 7 the analysis uses.
"""
import sys
import time

import d2_paths as P
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType

MASTER = "local[*]"
for a in sys.argv[1:]:
    if a.startswith("--master"):
        MASTER = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]

P.preflight()
SRC, DEST = P.POSTS_GLOB, P.TRIM_DIR

SCHEMA = StructType([
    StructField("id",          StringType(), True),
    StructField("author",      StringType(), True),
    StructField("subreddit",   StringType(), True),
    StructField("created_utc", LongType(),   True),   # long: epoch seconds, may be negative-safe
    StructField("score",       LongType(),   True),
    StructField("title",       StringType(), True),
    StructField("selftext",    StringType(), True),   # nested objects flatten, so keep as string
])

spark = P.spark("d2_trim", MASTER, mem="4g", shuffle="32")
t0 = time.time()

raw = spark.read.schema(SCHEMA).option("mode", "PERMISSIVE").json(SRC)
n_read = raw.count()

clean = (raw.dropna(subset=["id", "created_utc"])
            .withColumn("subreddit", F.trim("subreddit"))
            .withColumn("ts", F.to_timestamp(F.from_unixtime(F.col("created_utc"))))
            .withColumn("period",
                F.when(F.col("ts") < F.lit("2022-12-01 00:00:00"), F.lit("control_pre_LLM"))
                 .when(F.col("ts") < F.lit("2023-01-01 00:00:00"), F.lit("transition"))
                 .otherwise(F.lit("llm_era"))))

clean.repartition(8).write.mode("overwrite").parquet(DEST)
elapsed = round(time.time() - t0, 1)
print(f"\n[trim] wrote {DEST} in {elapsed} s")

# ---- read back: everything below is cheap ----
p = spark.read.parquet(DEST).cache()
n = p.count()

print("\n================ TABLE 6.1 / SECTION 4.2 CELLS ================")
print(f"  rows read from jsonl            : {n_read:,}")
print(f"  rows kept (non-null id + ts)    : {n:,}")
print(f"  rows dropped (null id/created_utc): {n_read - n:,}")
print(f"  trim time                       : {elapsed} s   (report separately in §7)")

print("\n================ TABLE 6.5 ROW COUNTS ================")
p.groupBy("period").count().orderBy("period").show(truncate=False)

print("================ TABLE 6.6: subreddit x period (need all 10 in BOTH) ================")
(p.groupBy("subreddit", "period").count().orderBy("subreddit", "period")
   .show(100, truncate=False))

print("========= CONTROL RANGE -> decides the Table 6.5 row LABEL =========")
(p.filter(F.col("period") == "control_pre_LLM")
   .select(F.min(F.date_format("ts", "yyyy-MM")).alias("first_month"),
           F.max(F.date_format("ts", "yyyy-MM")).alias("last_month"),
           F.count("*").alias("rows")).show(truncate=False))
(p.filter(F.col("period") == "llm_era")
   .select(F.min(F.date_format("ts", "yyyy-MM")).alias("first_month"),
           F.max(F.date_format("ts", "yyyy-MM")).alias("last_month"),
           F.count("*").alias("rows")).show(truncate=False))

print("================ SUBREDDIT STRINGS (10 expected) ================")
p.groupBy("subreddit").count().orderBy(F.desc("count")).show(30, truncate=False)

print(f"""
NEXT
  1. check 'rows kept' is plausible, and that every *_normal file landed in the control range
  2. then delete the .jsonl (and .zst) files - the Parquet has everything needed
  3. Table 6.1 row label: 'sample fraction: Dataset 2 = 1.0' (full data, no sampling)
""")
spark.stop()
