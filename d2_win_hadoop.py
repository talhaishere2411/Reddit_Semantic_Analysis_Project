"""Fix: java.lang.UnsatisfiedLinkError NativeIO$Windows.access0 on Windows.

WHY THIS HAPPENS
    Spark on Windows ships Hadoop's Java classes but NOT the native Windows shims
    (hadoop.dll, winutils.exe). Most operations fall back to Java implementations, but
    the GLOB path calls FileUtil.canRead -> NativeIO.Windows.access0, which is a native
    method. With no hadoop.dll there is nothing to bind to, so it throws, inside a
    ForkJoinPool worker thread - which is why the job looks "stuck" instead of erroring
    cleanly.

WHY DATASET 1 WORKED
    Dataset 1 read ONE file: spark.read.json("data/corpus-webis-tldr-17.json"). No glob,
    so RawLocalFileSystem.listStatus / Globber was never called. Dataset 2 passes
    data/*.jsonl - a wildcard - which forces exactly that code path.

FIX
    Put winutils.exe + hadoop.dll where Hadoop can find them and put that folder on
    PATH. Verified approach: download winutils for Hadoop 3.3.6 (the last build anyone
    publishes for Windows) even if your Spark bundles Hadoop 3.4/3.5 - these shims are
    a thin compatibility layer for local-filesystem calls only.

USAGE
    python d2_win_hadoop.py check     # what is installed, what is missing
    python d2_win_hadoop.py setup     # download into <pyspark>/hadoop/bin (no admin needed)
    python d2_win_hadoop.py test      # prove a glob read works, then report PASS/FAIL

After 'setup', every d2_*.py script picks it up automatically (d2_paths.spark() calls
win_env() first). No shell configuration, no admin rights, no reboot.
"""
import os
import platform
import shutil
import subprocess
import sys
import urllib.request

IS_WIN = platform.system() == "Windows"
VER = "3.3.6"
BASE = f"https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-{VER}/bin"
FILES = ["winutils.exe", "hadoop.dll"]


def spark_home():
    """<site-packages>/pyspark"""
    try:
        import pyspark
        return os.path.dirname(os.path.abspath(pyspark.__file__))
    except Exception:
        return None


def target_dir():
    sh = spark_home()
    return os.path.join(sh, "hadoop", "bin") if sh else None


def find_existing():
    """Return a directory that already has BOTH files, if any."""
    cands = []
    if os.environ.get("D2_HADOOP_BIN"):
        cands.append(os.environ["D2_HADOOP_BIN"])
    if os.environ.get("HADOOP_HOME"):
        cands.append(os.path.join(os.environ["HADOOP_HOME"], "bin"))
        cands.append(os.environ["HADOOP_HOME"])
    t = target_dir()
    if t:
        cands.append(t)
    if spark_home():
        cands.append(os.path.join(spark_home(), "bin"))
    cands += [r"C:\hadoop\bin", r"C:\hadoop", r"C:\winutils",
              r"C:\Program Files\hadoop\bin", r"C:\tools\hadoop\bin",
              os.path.expanduser(r"~\hadoop\bin"), os.path.expanduser(r"~\winutils")]
    for c in cands:
        if c and all(os.path.exists(os.path.join(c, f)) for f in FILES):
            return c
    return None


def win_env():
    """Set HADOOP_HOME and prepend to PATH in the CURRENT process. Called by
    d2_paths.spark() before the SparkSession is built, so the JVM inherits it."""
    if not IS_WIN:
        return False
    d = find_existing()
    if not d:
        return False
    hh = os.path.dirname(d) if os.path.basename(d) == "bin" else d
    os.environ["HADOOP_HOME"] = hh
    path = os.environ.get("PATH", "")
    if d.lower() not in path.lower():
        os.environ["PATH"] = d + os.pathsep + path
    return True


def java_library_path_hint(d):
    """For reference: what the JVM derives java.library.path from on Windows."""
    return d


def cmd_find():
    """Walk likely roots and report every winutils.exe / hadoop.dll found."""
    print("=" * 70)
    print("SEARCHING FOR winutils.exe / hadoop.dll")
    print("=" * 70)

    hits = {}
    w = shutil.which("winutils") or shutil.which("winutils.exe")
    if w:
        print(f"\n  on PATH: {w}")
        hits.setdefault(os.path.dirname(w), set()).add("winutils.exe")

    if os.environ.get("HADOOP_HOME"):
        hb = os.path.join(os.environ["HADOOP_HOME"], "bin")
        print(f"  HADOOP_HOME: {os.environ['HADOOP_HOME']}")
        for f in FILES:
            if os.path.exists(os.path.join(hb, f)):
                hits.setdefault(hb, set()).add(f)

    roots = ["C:\\", os.path.expanduser("~"),
             os.path.join(os.path.expanduser("~"), "Downloads"),
             os.path.join(os.path.expanduser("~"), "Documents")]
    SKIP = {"windows", "$recycle.bin", "program files", "program files (x86)",
            "programdata", "appdata", "node_modules", ".git", ".venv", "venv",
            "__pycache__", "anaconda3", "miniconda3", "temp", "tmp"}
    print("\n  scanning (bounded depth 4, skips system folders) ...")
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath[len(root):].count(os.sep)
            if depth > 4:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d.lower() not in SKIP]
            low = {f.lower() for f in filenames}
            for f in FILES:
                if f.lower() in low:
                    hits.setdefault(dirpath, set()).add(f)
                    print(f"    found {f:<14} in {dirpath}")
            if len(hits) > 12:
                print("    (stopping - many hits)")
                break

    print("\n" + "-" * 70)
    if not hits:
        print("  Neither file was found. Download both (need BOTH):")
        print(f"    {BASE}/winutils.exe")
        print(f"    {BASE}/hadoop.dll")
        print("  then:  python d2_win_hadoop.py setup")
        return
    both = {d: f for d, f in hits.items() if len(f) == 2}
    print("  USABLE directories (contain BOTH files):")
    for d in both:
        print(f"    {d}")
    partial = {d: f for d, f in hits.items() if len(f) < 2}
    if partial:
        print("\n  INCOMPLETE (only one of the two - hadoop.dll is the one that matters):")
        for d, f in partial.items():
            missing = [x for x in FILES if x not in f]
            print(f"    {d}   missing: {', '.join(missing)}")
    if both:
        best = sorted(both)[0]
        print(f"\n  NEXT -> point this project at it for the whole session:")
        print(f'    $env:D2_HADOOP_BIN = "{best}"')
        print(f"    .venv\\Scripts\\python.exe d2_win_hadoop.py test")
        print("\n  Or just copy the pair into the venv, which needs no env var at all:")
        print(f"    copy \"{best}\\*\" \"{target_dir() or '<pyspark>/hadoop/bin'}\"")
    print()


# ---------------------------------------------------------------- commands
def cmd_check():
    print("=" * 70)
    print("WINDOWS HADOOP NATIVE LIBRARY CHECK")
    print("=" * 70)
    sh = spark_home()
    print(f"\n  pyspark package : {sh or 'NOT IMPORTABLE'}")
    print(f"  target folder   : {target_dir() or '-'}")
    print(f"  HADOOP_HOME     : {os.environ.get('HADOOP_HOME', '<not set>')}")
    d = find_existing()
    print(f"\n  usable shim dir : {d or 'NOT FOUND  <-- this is why glob fails'}")
    if d:
        for f in FILES:
            p = os.path.join(d, f)
            print(f"    {f:<14} {os.path.getsize(p)/1024:8.1f} KB  {p}")
    else:
        print("\n  Run:  python d2_win_hadoop.py setup")
    print()


def cmd_setup():
    print("=" * 70)
    print(f"DOWNLOADING winutils {VER} shims (no admin rights required)")
    print("=" * 70)
    t = target_dir()
    if not t:
        raise SystemExit("  pyspark is not importable - install it first")
    os.makedirs(t, exist_ok=True)
    for f in FILES:
        url = f"{BASE}/{f}"
        dest = os.path.join(t, f)
        print(f"\n  {f}\n    from {url}\n    to   {dest}")
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            with open(dest, "wb") as fh:
                fh.write(data)
            print(f"    OK  {len(data)/1024:.1f} KB")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            print("\n  Manual route (2 minutes):")
            print(f"    1. open {BASE}")
            print("    2. download winutils.exe and hadoop.dll")
            print(f"    3. put both in: {t}")
            print("    4. re-run: python d2_win_hadoop.py test")
            return False
    print("\n  Downloaded. Now verify with:  python d2_win_hadoop.py test")
    return True


def cmd_test():
    print("=" * 70)
    print("FUNCTIONAL TEST: glob read through Hadoop native layer")
    print("=" * 70)
    if not win_env():
        print("\n  shims not found - run 'python d2_win_hadoop.py setup' first")
        return
    print(f"  HADOOP_HOME = {os.environ['HADOOP_HOME']}")
    print(f"  PATH prepend= {find_existing()}")
    import d2_paths as P
    print(f"  d2_paths version     = {getattr(P, 'VERSION', 'UNKNOWN - STALE COPY')}")
    files = P.list_jsonl()
    if not files:
        print(f"\n  no .jsonl in {P.DATA}/ - can still test the glob on the folder")
    if hasattr(P, "spark_glob"):
        glob_pattern = P.spark_glob()
    else:                                     # old d2_paths.py: compute it here
        print("  !! d2_paths.py is a stale copy (no spark_glob). Using a local fallback;")
        print("     replace d2_paths.py with the current version, then re-run.")
        d, f = os.path.split(P.POSTS_GLOB)
        glob_pattern = os.path.abspath(d).replace("\\", "/") + "/" + f
    print(f"\n  glob        = {glob_pattern}")
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master("local[2]").appName("win_test")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.driver.memory", "2g")
             .config("spark.sql.shuffle.partitions", "4")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    # Prove whether the JVM can actually SEE the shim folder. This is the single most
    # useful diagnostic: if 'C:\\hadoop\\bin' is absent here, the DLL cannot be loaded
    # no matter where the file sits on disk.
    try:
        jvm = spark.sparkContext._jvm
        libpath = jvm.System.getProperty("java.library.path")
        hh = jvm.System.getenv("HADOOP_HOME")
        print(f"\n  JVM HADOOP_HOME      = {hh}")
        print(f"  JVM java.library.path =")
        found = False
        for chunk in str(libpath).split(os.pathsep):
            mark = ""
            if any(f.lower() in str(chunk).lower() for f in ("hadoop", "winutils")):
                mark = "  <-- shim folder visible"
                found = True
            print(f"      {chunk}{mark}")
        if not found:
            print("      !! the shim folder is NOT on the JVM's java.library.path")
            print("         -> hadoop.dll cannot load; fix the environment, not the DLL")
    except Exception as e:
        print(f"  (could not read java.library.path: {type(e).__name__})")

    try:
        df = spark.read.json(glob_pattern)
        n = df.count()
        print(f"  read OK, rows = {n:,}")
        print(f"  columns      = {len(df.columns)}")
        print("\n  PASS - the glob works now. Re-run your trim.")
    except Exception as e:
        msg = str(e)
        print(f"  FAILED: {type(e).__name__}")
        print("  " + msg[:700].replace("\n", "\n  "))
        if "UnsatisfiedLinkError" in msg:
            print("\n  hadoop.dll is present but the JVM still cannot load it. Usually")
            print("  the MS Visual C++ redistributable is missing (msvcr120.dll /")
            print("  vcruntime140.dll). Install 'Microsoft Visual C++ Redistributable")
            print("  (x64)' then retry. Alternative: use d2_trim_pandas.py, which")
            print("  needs no Hadoop native layer at all.")
        elif "Path does not exist" in msg:
            print("\n  Different problem: the glob found nothing. Check the path printed")
            print("  above exists, and run: python d2_diag_input.py")
    finally:
        spark.stop()


if __name__ == "__main__":
    cmds = {"check": cmd_check, "find": cmd_find, "setup": cmd_setup, "test": cmd_test}
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    if what not in cmds:
        raise SystemExit("usage: python d2_win_hadoop.py {check|find|setup|test}")
    if not IS_WIN and what != "check":
        print("[note] not Windows - these shims are Windows-only; nothing to do.")
        raise SystemExit(0)
    cmds[what]()
