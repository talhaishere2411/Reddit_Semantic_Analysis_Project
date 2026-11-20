"""Verify your project folder holds ONE consistent set of files.

    python d2_selftest.py

Catches the failure mode we have now hit three times: some files replaced, others still
old, so a script calls a function that does not exist yet and dies deep into a run.

Needs no Spark, no Java, no data. Takes about a second.

Checks, per file:
  * present
  * syntax-valid (compiles)
  * exposes the symbols the other scripts depend on
  * reports the d2_paths generation, so a mixed folder is obvious
"""
import ast
import os
import sys

# file -> symbols other scripts rely on
REQUIRED = {
    "d2_paths.py":           ["VERSION", "DATA", "POSTS_GLOB", "TRIM_DIR", "OUT_D2",
                              "ensure_dirs", "ensure_all", "list_jsonl", "have_trim",
                              "posts_input", "spark_glob", "spark_dir", "preflight",
                              "spark"],
    "d2_prep_trim.py":       [],                       # script
    "d2_trim_pandas.py":     [],                       # script
    "d2_sanity_check.py":    [],
    "d2_verify_aggregation.py": [],
    "d2_threshold_sensitivity.py": [],
    "d2_charts.py":          [],
    "d2_validate_export_spark.py": [],
    "d2_fill_report_tables.py": [],
    "d2_diag_input.py":      [],
    "d2_win_hadoop.py":      ["find_existing", "win_env"],
    "env_check.py":          [],
    "check_vader_repro.py":  [],
    "venv_inspect.py":       [],
    "run_d2.py":             [],
}

OPTIONAL = {
    "requirements_locked.txt": "pinned versions from the Dataset 1 venv",
    "ENVIRONMENT.md": "venv / versions guide",
    "PATCH_AI_PIPELINE.md": "the 5 edits to ai_comments_pipeline.py",
    "REPORT_EDITS_POSTS_ONLY.md": "report edits removing the comments column",
    "BDA_DATASET2_PLAN.md": "the execution plan",
}

YOURS = ["ai_comments_pipeline.py", "sentiment_pipeline.py", "validate_sample.py",
         "ai_charts.py", "verify_aggregation.py", "make_charts.py", "scalability.py"]


def symbols(path):
    try:
        tree = ast.parse(open(path, encoding="utf8").read())
    except SyntaxError as e:
        return None, f"SyntaxError line {e.lineno}: {e.msg}"
    out = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return out, None


print("=" * 74)
print("PROJECT SELF-TEST  —  is this folder one consistent set of files?")
print("=" * 74)
print(f"  directory: {os.getcwd()}")

problems, ok_count = [], 0

print("\n[1] files and symbols")
for f, need in REQUIRED.items():
    if not os.path.exists(f):
        print(f"  MISSING   {f}")
        problems.append(f"{f} is missing")
        continue
    syms, err = symbols(f)
    if syms is None:
        print(f"  BROKEN    {f}   ({err})")
        problems.append(f"{f} does not parse: {err}")
        continue
    missing = [s for s in need if s not in syms]
    if missing:
        print(f"  STALE     {f}   missing: {', '.join(missing)}")
        problems.append(f"{f} is an old copy (missing {', '.join(missing)}) -> replace it")
    else:
        ok_count += 1
        extra = f"  ({len(need)} symbols ok)" if need else ""
        print(f"  ok        {f}{extra}")

print("\n[2] d2_paths generation")
ver = "?"
if os.path.exists("d2_paths.py"):
    try:
        sys.path.insert(0, os.getcwd())
        import d2_paths
        ver = getattr(d2_paths, "VERSION", "UNKNOWN - STALE COPY")
        print(f"  d2_paths.VERSION = {ver}")
        if ver == "UNKNOWN - STALE COPY":
            problems.append("d2_paths.py has no VERSION -> it is an old copy")
    except Exception as e:
        print(f"  cannot import d2_paths: {e}")
        problems.append(f"d2_paths.py not importable: {e}")

print("\n[3] your original project files")
for f in YOURS:
    if os.path.exists(f):
        src = open(f, encoding="utf8", errors="replace").read()
        notes = []
        if f == "ai_comments_pipeline.py":
            notes.append("makedirs" if "makedirs" in src else "NO makedirs <-- patch missing")
            notes.append("parquet-ok" if "read.parquet" in src else "NO parquet branch <-- patch missing")
        print(f"  present   {f:<30} {', '.join(notes)}")
    else:
        print(f"  absent    {f:<30} (fine if you do not need that stage)")

print("\n[4] inputs and outputs")
import glob as _g
n_jsonl = len(_g.glob(os.path.join("data", "*.jsonl")))
print(f"  data/*.jsonl        {n_jsonl}")
print(f"  data/posts_parquet  {'YES' if os.path.isdir(os.path.join('data','posts_parquet')) else 'not yet (trim has not run)'}")
for d in ("output", "output_ai", "spark_tmp"):
    print(f"  {d:<20}{'exists' if os.path.isdir(d) else 'created on demand'}")
if os.path.exists(os.path.join("output", "labelled.parquet")):
    print("  output/labelled.parquet  exists  -> check_vader_repro.py can run")
else:
    print("  output/labelled.parquet  MISSING -> copy it from the Dataset 1 project")


PIPELINE_CHECKS = [
    # (label, substring that must exist, why it matters)
    ("Edit 1  output dir created",        "makedirs",                     "pandas/matplotlib cannot create output_ai/"),
    ("Edit 2  parquet branch",            "read.parquet",                 "otherwise 9 GB schema inference per run"),
    ("Edit 3  null-ts rows dropped",      "isNotNull",                    "null created_utc would be labelled llm_era"),
    ("Edit 3b UTC session config",        "session.timeZone",             "period boundaries must not depend on the OS timezone"),
    ("Edit 5/7 driver memory 4g",         '"4g"',                         "2g dies on 2.89 M rows with a VADER UDF"),
    ("Edit 6  hadoop shims (win_env)",    "d2_win_hadoop",                "UnsatisfiedLinkError: NativeIO$Windows.access0"),
    ("Edit 8  PYSPARK_PYTHON pinned",     "PYSPARK_PYTHON",               "'Python worker failed to connect back' (Store stub / wrong interpreter)"),
]


def audit_pipeline():
    f = "ai_comments_pipeline.py"
    print("\n[6] ai_comments_pipeline.py — patch audit")
    if not os.path.exists(f):
        print("  absent (fine if you are not running the pipeline stage)")
        return []
    src = open(f, encoding="utf8", errors="replace").read()
    lines = src.splitlines()
    bad = []
    for label, needle, why in PIPELINE_CHECKS:
        ok = needle in src
        print(f"  {'ok      ' if ok else 'MISSING '} {label}")
        if not ok:
            print(f"           why it matters: {why}")
            bad.append(f"{label} -> {why}")

    # ORDERING: win_env must run BEFORE the JVM starts (SparkSession.builder)
    def first_line(*needles):
        for i, ln in enumerate(lines, 1):
            if any(n in ln for n in needles):
                return i
        return None

    hook = first_line("d2_win_hadoop", "HADOOP_HOME", "HADOOP_HOME")
    builder = first_line("SparkSession.builder")
    if hook and builder:
        if hook > builder:
            print(f"  BROKEN  ORDER: hadoop shim setup is on line {hook}, but "
                  f"SparkSession.builder is on line {builder}")
            print("           the JVM reads the environment at launch, so setting it")
            print("           afterwards has NO effect and the same")
            print("           UnsatisfiedLinkError will come back.")
            print("           -> move the try/import d2_win_hadoop block ABOVE the")
            print("              SparkSession.builder(...) call.")
            bad.append("Edit 6 is in the wrong place (after SparkSession.builder)")
        else:
            print(f"  ok       ORDER: shims set on line {hook}, before the session "
                  f"builder on line {builder}")
    elif not hook:
        bad.append("Edit 6 missing entirely")

    if "sys.argv[5]" in src:
        print("  ok       AI_THRESHOLD overridable from argv[5] (optional)")
    return bad

problems += audit_pipeline()

print("\n[5] optional docs")
for f, why in OPTIONAL.items():
    print(f"  {'ok      ' if os.path.exists(f) else 'absent  '} {f:<28} {why}")

print("\n" + "=" * 74)
if problems:
    print("VERDICT: FIX THESE FIRST")
    for p_ in problems:
        print("  -", p_)
    print("\n  Replace the listed files with the current versions, then re-run this test.")
    sys.exit(1)
print(f"VERDICT: OK - {ok_count} scripts consistent, d2_paths v{ver}")
print("  Next: python run_d2.py --stage smoke    (then: --stage full)")
print("=" * 74)
