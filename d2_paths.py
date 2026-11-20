"""Single source of truth for the flat project layout. Import this everywhere.

Layout assumed (everything in the root folder, no nested data/ai/posts/):

    BDA/
    ├── ai_comments_pipeline.py     # yours, printed in the report (1 file to patch)
    ├── sentiment_pipeline.py       # yours, Dataset 1, unchanged
    ├── validate_sample.py          # yours, used only for its 'score' step
    ├── d2_*.py                     # these helpers, drop-in
    ├── data/
    │   ├── corpus-webis-tldr-17.json      # Dataset 1
    │   ├── r_buildapc_posts.jsonl         # 20 jsonl files, flat, no subfolders
    │   ├── r_buildapc_posts_normal.jsonl
    │   └── posts_parquet/                 # created by d2_prep_trim.py
    ├── output/                     # created automatically (Dataset 1)
    ├── output_ai/                  # created automatically (Dataset 2)
    └── spark_tmp/                  # created automatically (Spark spill)
"""
import glob
import os
import platform
import shutil
import sys

# Bump this whenever any d2_*.py changes. Every script prints it, and d2_selftest.py
# compares it, so a mixed folder cannot go unnoticed.
VERSION = "2.3"

# ---------------------------------------------------------------- paths
DATA        = "data"
POSTS_GLOB  = os.path.join(DATA, "*.jsonl")     # flat: all 20 post files in data/
TRIM_DIR    = os.path.join(DATA, "posts_parquet")
OUT_D1      = "output"
OUT_D2      = "output_ai"
SPARK_TMP   = "spark_tmp"

# the subset of Arctic Shift fields the analysis uses (keeps schema inference off)
POST_FIELDS = {
    "id": "string",
    "author": "string",
    "subreddit": "string",
    "created_utc": "long",
    "score": "long",
    "title": "string",
    "selftext": "string",
}


# ---------------------------------------------------------------- dirs
def ensure_dirs(*names):
    """mkpath each name. Call this before ANY pandas to_csv / savefig / parquet write,
    because pandas and matplotlib (unlike Spark) do not create directories."""
    for n in names:
        if n:
            os.makedirs(n, exist_ok=True)
    return names


def ensure_all():
    ensure_dirs(OUT_D1, OUT_D2, SPARK_TMP)


# ---------------------------------------------------------------- inputs
def spark_glob(pattern=None):
    """The glob as Spark should receive it: ABSOLUTE, forward slashes.

    Spark resolves relative paths against the JVM's working directory, which is not
    always the shell's cwd - and on Windows a relative glob through Hadoop's Globber
    is a common source of 'Path does not exist' errors. An absolute path removes the
    whole class of problem.
    """
    p = pattern or POSTS_GLOB
    d, f = os.path.split(p)
    return os.path.abspath(d).replace("\\", "/") + "/" + f


def spark_dir(path=None):
    """Absolute forward-slash form of a directory, for spark.read.parquet()."""
    return os.path.abspath(path or TRIM_DIR).replace("\\", "/")


def list_jsonl(glob_pattern=None):
    """All post jsonl files in data/. Sorted so logs are reproducible."""
    return sorted(glob.glob(glob_pattern or POSTS_GLOB))


def have_trim():
    return os.path.isdir(TRIM_DIR) and bool(glob.glob(os.path.join(TRIM_DIR, "*.parquet")))


def posts_input():
    """What the pipeline / sanity check should read: trimmed Parquet if it exists,
    otherwise the raw jsonl glob. This is what makes the scripts 'just work' whether
    or not you have run the trim yet."""
    return TRIM_DIR if have_trim() else POSTS_GLOB


def preflight(verbose=True):
    """Fail loudly and early instead of dying 20 minutes into a Spark job."""
    problems = []
    if not os.path.isdir(DATA):
        problems.append(f"missing folder: {DATA}/ — create it and put the jsonl files inside")
    else:
        files = list_jsonl()
        if not files:
            problems.append(f"no *.jsonl matched {POSTS_GLOB} — the post files must sit "
                            f"directly in {DATA}/ (not in a subfolder)")
        if verbose:
            print(f"[paths] d2_paths v{VERSION}")
            print(f"[paths] cwd           = {os.getcwd()}")
            print(f"[paths] spark will read = {spark_glob()}")
            if " " in os.getcwd():
                print("[paths] WARNING: the working directory contains spaces. Spark on")
                print("                 Windows often fails to glob such paths. If Spark")
                print("                 reports 'Path does not exist', move the project to")
                print("                 a path without spaces (e.g. C:\\BDA) and retry.")
            gb = sum(os.path.getsize(f) for f in files) / 1e9
            print(f"[paths] {len(files)} jsonl file(s) in {DATA}/, {gb:.2f} GB total")
            for f in files[:30]:
                print("        ", os.path.basename(f), f"{os.path.getsize(f)/1e6:.0f} MB")
            if have_trim():
                print(f"[paths] trimmed Parquet present -> pipeline input = {TRIM_DIR}")
            else:
                print(f"[paths] no trim yet -> pipeline input = {POSTS_GLOB}")
    if platform.system() == "Windows" and verbose:
        stub = shutil.which("python")
        if stub and "windowsapps" in stub.lower():
            print("[paths] NOTE: bare 'python' is the Microsoft Store stub. Harmless for")
            print("               these scripts (d2_paths.spark() pins PYSPARK_PYTHON),")
            print("               but a plain Spark script outside this toolkit needs")
            print("               $env:PYSPARK_PYTHON = \"" + sys.executable + "\"")

    if problems:
        raise SystemExit("\n".join(["[paths] STOP:"] + ["  - " + p for p in problems]))
    return True


# ---------------------------------------------------------------- spark
def spark(app, master="local[*]", mem="4g", shuffle="16"):
    """One consistent session builder: UTC, fixed driver memory, spill dir inside the
    project so nothing lands in the OS temp folder."""
    from pyspark.sql import SparkSession

    # --- Python workers MUST be this venv's interpreter --------------------------
    # Spark launches workers by executing the bare command "python". On Windows that
    # resolves to the Microsoft Store app-execution-alias stub in ...\WindowsApps\,
    # which prints "Python was not found; run without arguments to install from the
    # Microsoft Store" and exits - so the worker never connects back and Spark reports
    #   SocketTimeoutException: Timed out while waiting for the Python worker to
    #   connect back
    # Pinning both variables to sys.executable guarantees (a) the worker starts and
    # (b) it can import vaderSentiment, which lives only in this venv.
    prev = os.environ.get("PYSPARK_PYTHON")
    if prev and os.path.abspath(prev) != os.path.abspath(sys.executable):
        print(f"[paths] overriding PYSPARK_PYTHON {prev} -> {sys.executable}")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    if platform.system() == "Windows":
        stub = shutil.which("python")
        if stub and "windowsapps" in stub.lower():
            print("[paths] note: the bare 'python' command on PATH is the Microsoft Store")
            print("               stub; PYSPARK_PYTHON now overrides it for Spark workers.")

    # Windows: Hadoop needs native shims (winutils.exe + hadoop.dll) for GLOB reads.
    # Without them a wildcard path dies with
    #   UnsatisfiedLinkError: NativeIO$Windows.access0
    # inside a ForkJoinPool worker, which looks like a hang. Applied here so every
    # script gets it without any shell setup.
    try:
        import d2_win_hadoop
        if d2_win_hadoop.win_env():
            print(f"[paths] hadoop shims: {os.environ.get('HADOOP_HOME')}")
    except Exception:
        pass

    ensure_all()
    s = (SparkSession.builder.master(master).appName(app)
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.driver.memory", mem)          # local mode: driver == executor
         .config("spark.sql.shuffle.partitions", shuffle)
         .config("spark.sql.files.maxPartitionBytes", "128m")
         .config("spark.local.dir", os.path.abspath(SPARK_TMP))
         .getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    return s
