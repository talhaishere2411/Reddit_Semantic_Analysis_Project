"""Extension: estimate the share of AI-like Reddit text (2023-2026) and compare its sentiment.
Usage: python ai_comments_pipeline.py <comments|posts> "<glob of .jsonl files>" [master] [fraction]
Input: Arctic Shift style JSON Lines (comments: body; posts: title + selftext)."""
import sys, time, json, os
from functools import reduce
from pyspark.sql import SparkSession, functions as F, types as T

# --- Windows: Hadoop needs winutils.exe + hadoop.dll for local-filesystem calls.
# Without this the JVM throws UnsatisfiedLinkError: NativeIO$Windows.access0 and
# spark.read.parquet / spark.read.json on a folder both fail. Must run before the
# SparkSession is created, because the JVM inherits the environment at launch.
# Spark's Python workers must be THIS venv: (a) the bare "python" on Windows is the
# Microsoft Store stub, which exits immediately -> "Python worker failed to connect
# back"; (b) vaderSentiment is installed only here.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
try:
    import d2_win_hadoop
    if d2_win_hadoop.win_env():
        print("[env] hadoop shims:", os.environ.get("HADOOP_HOME"))
except Exception:
    pass          # non-Windows, or the helper is absent: nothing to do

KIND   = sys.argv[1] if len(sys.argv) > 1 else "comments"
PATH   = sys.argv[2] if len(sys.argv) > 2 else "data/ai/comments/*.jsonl"
MASTER = sys.argv[3] if len(sys.argv) > 3 else "local[*]"
FRAC   = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
import os
AI_THRESHOLD = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0   # allows 2.5 / 3.5
MIN_CHARS = 40              # very short text cannot be assessed
OUT = "output_ai"
os.makedirs(OUT, exist_ok=True)          # <-- pandas/matplotlib need this

spark = (SparkSession.builder.master(MASTER).appName("AIcomments_" + KIND)
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.driver.memory", "4g")        # local mode: driver == executor
         .config("spark.local.dir", os.path.abspath("spark_tmp"))
         .getOrCreate())
os.makedirs("spark_tmp", exist_ok=True)
spark.sparkContext.setLogLevel("ERROR")
times = {}
def timed(name, fn):
    t = time.time(); r = fn(); times[name] = round(time.time() - t, 2); return r


# ---------- 1. Ingestion ----------
# Reads data/posts_parquet when present (fast, explicit schema), else the jsonl.
if PATH.rstrip("/").endswith(".parquet") or os.path.isdir(PATH) and "parquet" in PATH:
    raw = spark.read.parquet(PATH)
else:
    from pyspark.sql.types import StructType, StructField, StringType, LongType
    _SCHEMA = StructType([
        StructField("id", StringType(), True), StructField("author", StringType(), True),
        StructField("subreddit", StringType(), True), StructField("created_utc", LongType(), True),
        StructField("score", LongType(), True), StructField("title", StringType(), True),
        StructField("selftext", StringType(), True)])
    raw = spark.read.schema(_SCHEMA).option("mode", "PERMISSIVE").json(PATH)
if KIND == "comments":
    df = raw.select("id", "author", "subreddit", "created_utc", "score", F.col("body").alias("text"))
else:
    df = raw.select("id", "author", "subreddit", "created_utc", "score",
                    F.concat_ws(" ", F.col("title"), F.coalesce(F.col("selftext"), F.lit(""))).alias("text"))
if FRAC < 1.0:
    df = df.sample(False, FRAC, seed=42)
df = df.cache()
n_raw = timed("ingest_count", df.count)

# ---------- 2. Cleaning ----------
df = (df.dropna(subset=["id", "text"])
        .withColumn("text", F.trim("text"))
        .filter(~F.col("text").isin("[deleted]", "[removed]"))
        .filter(F.length("text") >= MIN_CHARS)
        .dropDuplicates(["id"])
                .withColumn("ts", F.to_timestamp(F.from_unixtime(F.col("created_utc").cast("long"))))
        .withColumn("month", F.date_format("ts", "yyyy-MM"))
        .withColumn("period", F.when(F.col("ts") < F.lit("2022-12-01 00:00:00"), F.lit("control_pre_LLM"))
                               .when(F.col("ts") < F.lit("2023-01-01 00:00:00"), F.lit("transition"))
                               .otherwise(F.lit("llm_era"))))
n_bad_ts = df.filter(F.col("ts").isNull()).count()      # would be mislabelled llm_era
print("rows dropped for null created_utc:", n_bad_ts)
df = df.filter(F.col("ts").isNotNull())
n_clean = timed("clean_count", df.count)

# ---------- 3. Automated accounts (analysed separately) ----------
low = F.lower("text")
auth = F.lower(F.coalesce(F.col("author"), F.lit("")))
df = df.withColumn("automated",
        (auth.rlike(r"(bot|automoderator)$") | low.rlike(r"i am a bot|this action was performed automatically")).cast("int"))

# ---------- 4. AI-likeness score (transparent heuristic, NOT ground truth) ----------
PHRASES = [r"\bdelve", r"\btapestry\b", r"it(?:'|\u2019)?s important to note", r"it is important to note",
           r"i hope this helps", r"\bgreat question\b", r"\bcertainly!", r"\babsolutely!", r"\bin conclusion\b",
           r"\bfurthermore\b", r"\bmoreover\b", r"navigating the", r"testament to", r"fast-paced world",
           r"it(?:'|\u2019)?s worth noting", r"plays a crucial role", r"\bmultifaceted\b", r"let me know if",
           r"feel free to", r"here are (?:some|a few)", r"key takeaways?", r"\bultimately,"]
phrase_hits = reduce(lambda a, b: a + b, [F.when(low.rlike(p), 1).otherwise(0) for p in PHRASES])
self_disc = low.rlike(r"as an ai\b|as a language model|i(?:'|\u2019)?m an ai\b|i am an ai\b|as an llm")
emdash_n = F.length("text") - F.length(F.regexp_replace("text", "\u2014", ""))
fmt_hits = (F.when(F.col("text").rlike(r"\*\*[^*]+\*\*"), 1).otherwise(0)
            + F.when(F.col("text").rlike(r"(?m)^\s*(?:[-*\u2022]|\d+\.)\s+\S"), 1).otherwise(0)
            + F.when(F.col("text").rlike(r"(?m)^#{1,3}\s"), 1).otherwise(0))
informal = low.rlike(r"\b(lol|lmao|tbh|idk|imo|imho|wtf|omg|ngl|smh)\b")
score = (F.when(self_disc, 3.0).otherwise(0.0)
         + F.least(phrase_hits, F.lit(3)).cast("double")
         + F.when(emdash_n >= 1, 1.0).otherwise(0.0)
         + F.when(fmt_hits >= 2, 1.0).otherwise(0.0)
         + F.when(F.length("text") >= 400, 0.5).otherwise(0.0)
         - F.when(informal, 1.0).otherwise(0.0))
df = (df.withColumn("ai_score", score)
        .withColumn("ai_like", ((F.col("ai_score") >= AI_THRESHOLD) & (F.col("automated") == 0)).cast("int")))

# ---------- 5. AI-accusation flag (paper-inspired lexicon, small subset) ----------
ACC = (r"ai[- ]slop|\bslop\b|chat ?gpt wrote|written (?:by|with) (?:an? )?(?:ai|chat ?gpt|llm)|ai[- ]generated|"
       r"is this (?:ai|chat ?gpt)|this is (?:ai|chat ?gpt)|op is a bot|(?:sounds|reads) like (?:chat ?gpt|an? ai|an llm)|\bllm\b")
df = df.withColumn("accusation", low.rlike(ACC).cast("int"))

# ---------- 6. VADER sentiment (same method and thresholds as the first dataset) ----------
_an = None
def vader_compound(t):
    global _an
    if _an is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _an = SentimentIntensityAnalyzer()
    return float(_an.polarity_scores(t[:1000])["compound"])
vader_udf = F.udf(vader_compound, T.DoubleType())
df = (df.withColumn("compound", vader_udf("text"))
        .withColumn("label", F.when(F.col("compound") >= 0.05, "positive")
                              .when(F.col("compound") <= -0.05, "negative").otherwise("neutral"))).cache()
timed("score_and_label", df.count)
df.drop("author", "text").write.mode("overwrite").parquet(f"{OUT}/{KIND}_scored.parquet")   # no usernames or raw text stored

# ---------- 7. Aggregations (Spark SQL) ----------
df.createOrReplaceTempView("t")
Q = {
 "share_by_period": """SELECT period, COUNT(*) n, SUM(ai_like) ai_like_n, ROUND(100*AVG(ai_like),3) ai_like_pct,
                              ROUND(100*AVG(accusation),3) accusation_pct, ROUND(100*AVG(automated),3) automated_pct
                       FROM t GROUP BY period ORDER BY period""",
 "share_by_month": """SELECT month, COUNT(*) n, ROUND(100*AVG(ai_like),3) ai_like_pct, ROUND(100*AVG(accusation),3) accusation_pct
                      FROM t GROUP BY month ORDER BY month""",
 "share_by_subreddit": """SELECT subreddit, period, COUNT(*) n, ROUND(100*AVG(ai_like),3) ai_like_pct
                          FROM t GROUP BY subreddit, period HAVING COUNT(*) >= 200 ORDER BY subreddit, period""",
 "sentiment_ai_vs_human": """SELECT period, ai_like, COUNT(*) n, ROUND(AVG(compound),4) avg_compound,
                                    ROUND(STDDEV(compound),4) sd_compound,
                                    ROUND(100*AVG(CASE WHEN label='positive' THEN 1 ELSE 0 END),2) pos_pct,
                                    ROUND(100*AVG(CASE WHEN label='neutral'  THEN 1 ELSE 0 END),2) neu_pct,
                                    ROUND(100*AVG(CASE WHEN label='negative' THEN 1 ELSE 0 END),2) neg_pct,
                                    ROUND(AVG(score),2) avg_upvotes
                             FROM t WHERE automated = 0 GROUP BY period, ai_like ORDER BY period, ai_like""",
 "sentiment_accusations": """SELECT period, accusation, COUNT(*) n, ROUND(AVG(compound),4) avg_compound,
                                    ROUND(100*AVG(CASE WHEN label='negative' THEN 1 ELSE 0 END),2) neg_pct
                             FROM t WHERE automated = 0 GROUP BY period, accusation ORDER BY period, accusation""",
}
def save_all():
    for name, sql in Q.items():
        spark.sql(sql).toPandas().to_csv(f"{OUT}/{KIND}_{name}.csv", index=False)
timed("aggregations", save_all)

summary = {"kind": KIND, "master": MASTER, "fraction": FRAC, "ai_threshold": AI_THRESHOLD,
           "rows_raw": n_raw, "rows_clean": n_clean, "retained_pct": round(100 * n_clean / max(n_raw, 1), 2),
           "stage_seconds": times}
json.dump(summary, open(f"{OUT}/{KIND}_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
spark.stop()
