"""Diagnose 'Path does not exist' / Hadoop Globber errors before Spark is involved.

    python d2_diag_input.py

Prints: the working directory, the exact glob Spark will receive, what Python's own
glob finds, and then tries one tiny Spark read with an ABSOLUTE path. Every step is
labelled, so whichever line fails is the culprit.

Read-only. Touches no data, writes nothing.
"""
import glob
import json
import os
import sys

import d2_paths as P

print("=" * 74)
print("INPUT DIAGNOSTIC")
print("=" * 74)

# ---------------------------------------------------------------- 1. where are we
print("\n[1] working directory")
cwd = os.getcwd()
print(f"  os.getcwd()      {cwd}")
print(f"  spaces in path?  {'YES  <-- known Windows troublemaker for Spark/Hadoop' if ' ' in cwd else 'no'}")
print(f"  non-ascii?       {'YES' if any(ord(c) > 127 for c in cwd) else 'no'}")

# ---------------------------------------------------------------- 2. relative
print("\n[2] what the scripts ask for (relative)")
print(f"  P.POSTS_GLOB     {P.POSTS_GLOB}")
rel = sorted(glob.glob(P.POSTS_GLOB))
print(f"  python glob      {len(rel)} files")
for f in rel[:5]:
    print("      ", f)
if len(rel) > 5:
    print(f"       ... and {len(rel)-5} more")

# ---------------------------------------------------------------- 3. absolute
print("\n[3] what Spark will ACTUALLY be given (absolute, forward slashes)")
abs_glob = P.spark_glob()
print(f"  spark input      {abs_glob}")
absf = sorted(glob.glob(abs_glob))
print(f"  python glob      {len(absf)} files")

# ---------------------------------------------------------------- 4. the dir
print("\n[4] the data directory")
d = os.path.abspath(P.DATA)
print(f"  {d}   exists={os.path.isdir(d)}")
if os.path.isdir(d):
    entries = os.listdir(d)
    js = [e for e in entries if e.endswith(".jsonl")]
    other = [e for e in entries if not e.endswith(".jsonl")]
    print(f"  .jsonl files     {len(js)}")
    print(f"  other entries    {len(other)}  -> {other[:8]}")
    if not js:
        print("  !! no .jsonl directly in data/ - are they in a subfolder?")
        for sub in other:
            sp = os.path.join(d, sub)
            if os.path.isdir(sp):
                n = len([x for x in os.listdir(sp) if x.endswith(".jsonl")])
                if n:
                    print(f"     subfolder '{sub}' contains {n} .jsonl files")

# ---------------------------------------------------------------- 5. parse one
print("\n[5] can one record actually be parsed?")
SCHEMA = None
try:
    from pyspark.sql.types import StructType, StructField, StringType, LongType
    SCHEMA = StructType([
        StructField("id", StringType(), True), StructField("author", StringType(), True),
        StructField("subreddit", StringType(), True), StructField("created_utc", LongType(), True),
        StructField("score", LongType(), True), StructField("title", StringType(), True),
        StructField("selftext", StringType(), True)])
except ImportError:
    print("  (pyspark not importable here - steps 5-6 skipped, but 1-4 still valid)")
if absf:
    try:
        with open(absf[0], encoding="utf8", errors="replace") as fh:
            rec = json.loads(fh.readline())
        print(f"  file             {os.path.basename(absf[0])}")
        print(f"  size             {os.path.getsize(absf[0])/1e6:.1f} MB")
        print(f"  keys in record 1 {len(rec)}")
        for k in ("id", "created_utc", "subreddit", "title", "selftext"):
            print(f"    {k:<14} {'present: ' + repr(rec.get(k))[:60] if k in rec else 'MISSING'}")
        bad = [k for k in ("id", "created_utc") if rec.get(k) is None]
        if bad:
            print(f"  !! null in first record: {bad}")
    except Exception as e:
        print(f"  !! cannot parse {absf[0]}: {type(e).__name__}: {e}")

# ---------------------------------------------------------------- 6. tiny spark read
print("\n[6] tiny Spark read on ONE file, absolute path")
if SCHEMA is None:
    print("  skipped (no pyspark)")
    raise SystemExit(0)
try:
    spark = P.spark("d2_diag", "local[2]", mem="2g", shuffle="4")
    one = absf[0].replace("\\", "/") if absf else None
    print(f"  reading          file:///{one.lstrip('/')}" if one else "  no file to read")
    df = spark.read.schema(SCHEMA).option("mode", "PERMISSIVE").json(f"file:///{one.lstrip('/')}")
    print(f"  rows             {df.count():,}")
    df.select("id", "subreddit", "created_utc").show(3, truncate=False)
    print("  => Spark CAN read this. The glob pattern is the problem, not the data.")
    spark.stop()
except Exception as e:
    msg = str(e)[:600]
    print(f"  !! SPARK FAILED: {type(e).__name__}")
    print("  " + msg.replace("\n", "\n  "))
    print("\n  Common fixes, in order of likelihood on Windows:")
    print("   1. the working directory contains SPACES ('college things', 'BDA Project').")
    print("      Move the project to C:\\BDA and retry - this is the usual culprit.")
    print("   2. pass an absolute path with forward slashes (already done above; if this")
    print("      still fails, it is #1).")
    print("   3. a stray non-.jsonl or corrupt file in data/ - see step [4].")

print("\n" + "=" * 74)
