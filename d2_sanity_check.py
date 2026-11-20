"""Dataset 2 sanity check: proves the inputs are what the report claims, before the
long run. Reads the trimmed Parquet if it exists, otherwise the raw jsonl in data/.

    python d2_sanity_check.py

Checks
  1. columns present, and WHICH key holds the post time
  2. per-file period verdict: every *_normal.jsonl must be 2018-2020 / control_pre_LLM
  3. null-timestamp rows  -> these silently become 'llm_era' and corrupt Tables 6.5/6.6/7.4
  4. subreddit strings (must equal Dataset 1's exactly)
  5. null rates on keys (Table 7.1 'before cleaning')
  6. raw totals for Table 6.1, and the observed control month range
"""
import glob
import os

import d2_paths as P
from pyspark.sql import functions as F

CONTROL_YEARS = {2018, 2019, 2020}      # accepted control-period tolerance

P.preflight()
PATH = P.spark_dir() if P.have_trim() else P.spark_glob()
print(f"[sanity] reading {PATH}")

spark = P.spark("d2_sanity")

if P.have_trim():
    raw = spark.read.parquet(PATH)
else:
    from pyspark.sql.types import StructType, StructField, StringType, LongType
    SCHEMA = StructType([
        StructField("id", StringType(), True), StructField("author", StringType(), True),
        StructField("subreddit", StringType(), True), StructField("created_utc", LongType(), True),
        StructField("score", LongType(), True), StructField("title", StringType(), True),
        StructField("selftext", StringType(), True)])
    raw = spark.read.schema(SCHEMA).option("mode", "PERMISSIVE").json(PATH)

# ---------------------------------------------------------------- 1. columns
print("\n=== 1. columns present ===")
print(sorted(raw.columns))
for req in ("id", "subreddit", "created_utc"):
    if req not in raw.columns:
        raise SystemExit(f"[sanity] STOP: missing required column '{req}'")

print("\n=== 1b. which key holds the POST time? ===")
for k in ("created_utc", "created", "author_created_utc", "retrieved_on"):
    if k in raw.columns:
        r = raw.select(F.count(F.col(k)).alias("non_null"),
                       F.sum(F.col(k).isNull().cast("int")).alias("nulls"),
                       F.min(F.col(k)).alias("lo"), F.max(F.col(k)).alias("hi")).first()
        print(f"  {k:20} non_null={r['non_null']:>12,}  null={r['nulls']:>10,}  "
              f"epoch {r['lo']} .. {r['hi']}")
print("  ^ created_utc is the only field that may drive the period. author_created_utc is")
print("    the AUTHOR'S ACCOUNT creation date - it looks plausible while being wrong.")
if raw.filter(F.col("created_utc").isNull()).count() and PATH != P.TRIM_DIR:
    print("  !! null created_utc present; the trim drops those rows, so run d2_prep_trim.py")

# ---------------------------------------------------------------- 2. period
df = (raw.withColumn("ts", F.to_timestamp(F.from_unixtime(F.col("created_utc").cast("long"))))
         .withColumn("period",
             F.when(F.col("ts") < F.lit("2022-12-01 00:00:00"), F.lit("control_pre_LLM"))
              .when(F.col("ts") < F.lit("2023-01-01 00:00:00"), F.lit("transition"))
              .otherwise(F.lit("llm_era")))
         .withColumn("_year", F.year("ts"))
         .withColumn("_month", F.date_format("ts", "yyyy-MM")))

if "_file" not in df.columns:                # parquet has no input_file_name
    df = df.withColumn("_file", F.lit("(trimmed parquet)"))

print("\n=== 2. period x year ===")
df.groupBy("period", "_year").count().orderBy("period", "_year").show(50, truncate=False)

# ---------------------------------------------------------------- 3. null ts
print("=== 3. null-timestamp rows (would be mislabelled 'llm_era') ===")
n_null = df.filter(F.col("ts").isNull()).count()
print(f"  null-ts rows: {n_null:,}")
if n_null:
    print("  !! These are labelled llm_era by the when()/otherwise() chain. If using raw")
    print("     jsonl, run d2_prep_trim.py (it drops them and reports how many). Otherwise")
    print("     fix the source key before trusting Table 6.5.")

# ---------------------------------------------------------------- 4. subreddits
print("\n=== 4. subreddit strings (10 expected, must match Dataset 1) ===")
df.groupBy("subreddit").count().orderBy(F.desc("count")).show(30, truncate=False)
print("  distinct:", df.select("subreddit").distinct().count())

# ---------------------------------------------------------------- 5. nulls
print("=== 5. null rates on keys (Table 7.1 'before cleaning') ===")
(df.select([F.round(100 * F.avg(F.col(c).isNull().cast("int")), 3).alias("null_pct_" + c)
            for c in ("id", "subreddit", "created_utc")]).show())

# ---------------------------------------------------------------- 6. totals
print("=== 6. totals + observed control range (Table 6.1, Table 6.5 label) ===")
print(f"  total rows      : {df.count():,}")
print(f"  control_pre_LLM : {df.filter(F.col('period') == 'control_pre_LLM').count():,}")
print(f"  llm_era         : {df.filter(F.col('period') == 'llm_era').count():,}")
df.filter(F.col("period") == "control_pre_LLM").select(
    F.min("_month").alias("control_first"), F.max("_month").alias("control_last")).show()
print("  if that range is not 2019-01..2019-12, relabel the Table 6.5 row to the real range.")
spark.stop()
