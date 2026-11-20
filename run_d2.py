"""One command for the whole Dataset 2 run. Flat layout, all output dirs auto-created.

    python run_d2.py                 # trim (if needed) -> sanity -> smoke -> full run
                                     # -> verify -> thresholds -> charts -> report dump
    python run_d2.py --stage full     # resume from a stage
    python run_d2.py --skip-trim      # you already have data/posts_parquet
    python run_d2.py --smoke-only     # just check the wiring on 2% of the data

Stages
    preflight : data/*.jsonl present, sizes printed
    trim      : d2_prep_trim.py            (the only 10 GB pass; skipped if parquet exists)
    sanity    : d2_sanity_check.py         (gate: control files must be 2018-2020)
    smoke     : ai_comments_pipeline 2%    (gate: does the pipeline actually run)
    full      : ai_comments_pipeline 1.0
    verify    : d2_verify_aggregation.py   (Table 7.1 row 2)
    thresh    : d2_threshold_sensitivity.py
    charts    : d2_charts.py               (Figures 6.3-6.5)
    tables    : d2_fill_report_tables.py   (prints every yellow cell)

Requires ONE hand-edit first: ai_comments_pipeline.py must be able to read the Parquet
and create output_ai/. See PATCH_AI_PIPELINE.md. This runner checks for that and stops
with a clear message if it is missing.
"""
import os
import subprocess
import sys
import time

import d2_paths as P

PY = sys.executable
STAGES = ["preflight", "trim", "sanity", "smoke", "full", "verify", "thresh", "charts", "tables"]

args = sys.argv[1:]
START = "preflight"
SKIP_TRIM = P.have_trim()
SMOKE_ONLY = "--smoke-only" in args
for a in args:
    if a.startswith("--stage"):
        START = a.split("=", 1)[1] if "=" in a else args[args.index(a) + 1]
    if a == "--skip-trim":
        SKIP_TRIM = True
if START not in STAGES:
    raise SystemExit(f"--stage must be one of {STAGES}")

P.ensure_all()          # output/ , output_ai/ , spark_tmp/  created here, once


def run(cmd, label, critical=True):
    print(f"\n{'='*78}\n>>> {label}\n>>> {' '.join(cmd)}\n{'='*78}", flush=True)
    t = time.time()
    rc = subprocess.call(cmd)
    print(f"<<< {label} finished in {round(time.time()-t,1)} s (exit {rc})", flush=True)
    if rc != 0 and critical:
        raise SystemExit(f"[run_d2] STOP: stage '{label}' failed with exit {rc}")
    return rc


def pipeline(frac):
    return [PY, "ai_comments_pipeline.py", "posts", P.TRIM_DIR, "local[*]", str(frac)]


# ---------------------------------------------------------------- gate: patched?
# Only the stages that RUN your pipeline need the patch. trim/sanity/verify/thresh/
# charts/tables all work on Parquet and are safe to run before you edit it.
NEEDS_PATCH = {"smoke", "full"}
pipe_src = open("ai_comments_pipeline.py").read() if os.path.exists("ai_comments_pipeline.py") else ""
if not pipe_src and (set(STAGES[STAGES.index(START):]) & NEEDS_PATCH):
    raise SystemExit("[run_d2] STOP: ai_comments_pipeline.py not found in this folder")
_unpatched = [m for m, ok in (
    ("output dir creation  (os.makedirs(OUT))", "makedirs" in pipe_src),
    ("parquet-aware reader (spark.read.parquet)", ("read.parquet" in pipe_src) or ("read_posts" in pipe_src)),
) if not ok]
if _unpatched and (set(STAGES[STAGES.index(START):]) & NEEDS_PATCH):
    raise SystemExit("[run_d2] STOP: ai_comments_pipeline.py is missing:\n  - "
                     + "\n  - ".join(_unpatched)
                     + "\nApply PATCH_AI_PIPELINE.md first (5 small edits).\n"
                     "NOTE: you can still run the pre-pipeline stages now:\n"
                     "      python run_d2.py --stage trim")

# ---------------------------------------------------------------- stages
for stage in STAGES[STAGES.index(START):]:
    if stage == "preflight":
        P.preflight()

    elif stage == "trim":
        if SKIP_TRIM:
            print(f"\n[run_d2] trim skipped — using {P.TRIM_DIR}")
        else:
            run([PY, "d2_prep_trim.py"], "trim jsonl -> parquet")

    elif stage == "sanity":
        run([PY, "d2_sanity_check.py"], "sanity check")

    elif stage == "smoke":
        run(pipeline(0.02), "smoke run (2%)")
        if SMOKE_ONLY:
            print("\n[run_d2] --smoke-only: stopping here. Wiring is fine.")
            break

    elif stage == "full":
        run(pipeline(1.0), "FULL RUN")

    elif stage == "verify":
        run([PY, "d2_verify_aggregation.py"], "aggregation correctness")

    elif stage == "thresh":
        run([PY, "d2_threshold_sensitivity.py"], "threshold sensitivity")

    elif stage == "charts":
        run([PY, "d2_charts.py"], "figures")

    elif stage == "tables":
        run([PY, "d2_fill_report_tables.py"], "report cells")

print("""
================================================================================
NEXT, BY HAND
  1. python d2_validate_export_spark.py 200     -> output_ai/validation_sample.csv
  2. label every row AI or HUMAN (sort by id first, hide ai_like)
  3. python validate_sample.py score            -> Table 7.4 last two rows
  4. write the 3 prose pieces: section 6.5, 6.6, and 7
  5. if the trim worked, delete data/*.jsonl and data/*.zst (~10 GB back)
================================================================================
""")
