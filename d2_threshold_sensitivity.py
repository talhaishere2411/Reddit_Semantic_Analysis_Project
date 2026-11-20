"""Threshold sensitivity of the AI-likeness heuristic (Table 7.1, last row).

No pipeline re-run needed: ai_score and automated are persisted, and
ai_like = (ai_score >= t) AND (automated == 0), so this is a 20-second recomputation.

    python d2_threshold_sensitivity.py            # 2.5 / 3.0 / 3.5
    python d2_threshold_sensitivity.py 2.5 2.7 3.0 3.3 3.5
"""
import os
import sys

import d2_paths as P
import pandas as pd
from pyspark.sql import functions as F

KIND = "posts"
THRESHOLDS = [float(x) for x in sys.argv[1:] if x.replace(".", "").isdigit()] or [2.5, 3.0, 3.5]
src = os.path.join(P.OUT_D2, f"{KIND}_scored.parquet")
if not os.path.isdir(src):
    raise SystemExit(f"[threshold] missing {src} — run ai_comments_pipeline.py first")

spark = P.spark("d2_threshold")
p = spark.read.parquet(src).cache()
print(f"[threshold] scored rows: {p.count():,}")

rows = []
for t in THRESHOLDS:
    x = (p.withColumn("flag", ((F.col("ai_score") >= t) & (F.col("automated") == 0)).cast("int"))
           .groupBy("period")
           .agg(F.count("*").alias("n"), F.avg("flag").alias("pct"), F.sum("flag").alias("flag_n"))
           .toPandas())
    d = {r.period: r for r in x.itertuples()}
    c, e = d.get("control_pre_LLM"), d.get("llm_era")
    cp = round(100 * c.pct, 3) if c is not None else None
    ep = round(100 * e.pct, 3) if e is not None else None
    rows.append({"threshold": t, "control_pct": cp, "llm_era_pct": ep,
                 "excess_pp": round(ep - cp, 3) if None not in (cp, ep) else None,
                 "control_n": int(c.n) if c is not None else None,
                 "llm_era_flagged_n": int(e.flag_n) if e is not None else None,
                 "llm_era_n": int(e.n) if e is not None else None})

r = pd.DataFrame(rows)
P.ensure_dirs(P.OUT_D2)
dest = os.path.join(P.OUT_D2, f"{KIND}_threshold_sensitivity.csv")
r.to_csv(dest, index=False)
print(r.to_string(index=False))
print(f"\n[threshold] wrote {dest}")
print("Report whether the excess is stable across 2.5-3.5. If it changes sign, the")
print("threshold is doing too much work - that is a limitation to state, not hide.")
spark.stop()
