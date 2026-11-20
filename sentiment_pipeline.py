import sys, time, json
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import (RegexTokenizer, StopWordsRemover,
                                CountVectorizer, IDF, StringIndexer)
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

MASTER = sys.argv[1] if len(sys.argv) > 1 else "local[*]"
FRAC   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
PATH   = sys.argv[3] if len(sys.argv) > 3 else "data/corpus-webis-tldr-17.json"
MINPOSTS = int(sys.argv[4]) if len(sys.argv) > 4 else 500
OUT    = "output"

spark = (SparkSession.builder.master(MASTER).appName("RedditSentiment")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.driver.memory", "2g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
times = {}

def timed(name, fn):
    t = time.time(); r = fn(); times[name] = round(time.time() - t, 2); return r

# 1. Ingestion
raw = spark.read.json(PATH).select("id", "subreddit", "summary", "content")
if FRAC < 1.0:
    raw = raw.sample(False, FRAC, seed=42)
raw = raw.cache()
n_raw = timed("ingest_count", raw.count)

# 2. Cleaning
df = raw.dropna(subset=["id", "summary"])
df = df.withColumn("summary", F.trim("summary")).filter(F.length("summary") >= 10)
df = df.dropDuplicates(["id"])
txt = F.regexp_replace("summary", r"http\S+", " ")
txt = F.regexp_replace(txt, r"[^a-zA-Z\s']", " ")
df = df.withColumn("clean", F.lower(txt)).withColumn("summary_len", F.length("summary"))
n_clean = timed("clean_count", df.count)

# 3. VADER labelling
_an = None
def vader_compound(text):
    global _an
    if _an is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _an = SentimentIntensityAnalyzer()
    return float(_an.polarity_scores(text)["compound"])

vader_udf = F.udf(vader_compound, T.DoubleType())
df = (df.withColumn("compound", vader_udf("summary"))
        .withColumn("label", F.when(F.col("compound") >= 0.05, "positive")
                              .when(F.col("compound") <= -0.05, "negative")
                              .otherwise("neutral"))).cache()
timed("vader_label", df.count)
df.write.mode("overwrite").parquet(f"{OUT}/labelled.parquet")

# 4. Aggregations (Spark SQL)
df.createOrReplaceTempView("posts")
dist = spark.sql("SELECT label, COUNT(*) n FROM posts GROUP BY label ORDER BY n DESC")
sub = spark.sql(f"""SELECT subreddit, COUNT(*) n, ROUND(AVG(compound),4) avg_compound
                    FROM posts GROUP BY subreddit HAVING COUNT(*) >= {MINPOSTS}
                    ORDER BY avg_compound DESC""")
len_by = spark.sql("SELECT label, ROUND(AVG(summary_len),1) avg_len FROM posts GROUP BY label")
def save_aggs():
    dist.toPandas().to_csv(f"{OUT}/dist.csv", index=False)
    sub.toPandas().to_csv(f"{OUT}/subreddit.csv", index=False)
    len_by.toPandas().to_csv(f"{OUT}/len_by_label.csv", index=False)
timed("aggregations", save_aggs)

# 5. MLlib classification
train, test = df.randomSplit([0.8, 0.2], seed=42)
acc = MulticlassClassificationEvaluator(labelCol="y", metricName="accuracy")
f1  = MulticlassClassificationEvaluator(labelCol="y", metricName="f1")
metrics = []
for vocab in (3000, 5000):
    for name in ("logreg", "naive_bayes"):
        stages = [StringIndexer(inputCol="label", outputCol="y"),
                  RegexTokenizer(inputCol="clean", outputCol="tok", pattern="\\s+"),
                  StopWordsRemover(inputCol="tok", outputCol="words"),
                  CountVectorizer(inputCol="words", outputCol="tf", vocabSize=vocab, minDF=5)]
        if name == "logreg":
            stages += [IDF(inputCol="tf", outputCol="features"),
                       LogisticRegression(labelCol="y", maxIter=20, regParam=0.01)]
        else:
            stages += [NaiveBayes(labelCol="y", featuresCol="tf", smoothing=1.0)]
        t = time.time()
        model = Pipeline(stages=stages).fit(train)
        fit_t = round(time.time() - t, 2)
        pred = model.transform(test)
        metrics.append({"model": name, "vocab": vocab, "accuracy": round(acc.evaluate(pred), 4),
                        "weighted_f1": round(f1.evaluate(pred), 4), "train_seconds": fit_t})

# 6. Save run summary
tag = MASTER.replace("[", "").replace("]", "").replace("*", "all")
summary = {"master": MASTER, "fraction": FRAC, "rows_raw": n_raw, "rows_clean": n_clean,
           "retained_pct": round(100 * n_clean / max(n_raw, 1), 2),
           "stage_seconds": times, "ml": metrics}
json.dump(summary, open(f"{OUT}/summary_{tag}_{FRAC}.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
spark.stop()
