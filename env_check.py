"""Print the exact runtime versions and compare them with what the report claims.

    python env_check.py            (with the venv active, from the project root)

Why bother: Table 6.9 and Table 8.1 of the report state Spark 4.3.0, Python 3.13.4,
Java 25.0.1. Those claims must match the machine that produced the numbers - and the
Dataset 1 and Dataset 2 results are only comparable if VADER is the same version,
because its lexicon ships inside the package.

This output is also a good screenshot for section 7 "Test setup" / Table 8.1 (the
report's page-1 note asks for screenshots).
"""
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata as _md

EXPECTED = {"python": "3.13.4", "spark": "4.3.0", "java": "25.0.1"}


def line(k, v, note=""):
    print(f"  {k:<22} {v}{('   ' + note) if note else ''}")


def sh(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (out.stdout or out.stderr).strip()
    except Exception as e:
        return f"<unavailable: {e}>"


print("=" * 72)
print("ENVIRONMENT CHECK  —  BDA Reddit sentiment project")
print("=" * 72)

# ---------------------------------------------------------------- python
in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
print("\n[interpreter]")
line("python", platform.python_version(),
     "OK" if platform.python_version() == EXPECTED["python"] else f"(report says {EXPECTED['python']})")
line("executable", sys.executable)
line("running in a venv", "YES" if in_venv else "NO  <-- use .venv\\Scripts\\python.exe")
line("venv path", sys.prefix if in_venv else "-")
line("platform", f"{platform.system()} {platform.release()} ({platform.machine()})")

# ---------------------------------------------------------------- java
print("\n[java / hadoop]")
java = sh(["java", "-version"])
first = java.splitlines()[0] if java else "<none>"
line("java -version", first,
     "OK" if EXPECTED["java"] in first else f"(report says {EXPECTED['java']})")
line("JAVA_HOME", os.environ.get("JAVA_HOME", "<not set - Spark needs it>"))
line("SPARK_HOME", os.environ.get("SPARK_HOME", "<not set - fine for pip pyspark>"))
line("HADOOP_HOME", os.environ.get("HADOOP_HOME", "<not set>"))
if platform.system() == "Windows":
    winutils = shutil.which("winutils") or (
        os.path.join(os.environ.get("HADOOP_HOME", ""), "bin", "winutils.exe")
        if os.environ.get("HADOOP_HOME") else None)
    ok = bool(winutils and os.path.exists(str(winutils)))
    line("winutils.exe", str(winutils) if ok else "<not found>",
         "" if ok else "(only needed if a parquet write fails with UnsatisfiedLinkError)")

# ---------------------------------------------------------------- packages
print("\n[python packages]")
PKGS = ["pyspark", "vaderSentiment", "pandas", "numpy", "pyarrow",
        "matplotlib", "zstandard"]
mods = {"pyspark": "pyspark", "vaderSentiment": "vaderSentiment", "pandas": "pandas",
        "numpy": "numpy", "pyarrow": "pyarrow", "matplotlib": "matplotlib",
        "zstandard": "zstandard"}


def pkg_version(dist_name, module=None):
    """Many packages (vaderSentiment) expose no __version__; ask the metadata instead.
    Three fallbacks, because dist names are not always import names."""
    for name in (dist_name, module or dist_name):
        try:
            return _md.version(name)
        except Exception:
            pass
    try:                                   # scan installed dists, case-insensitive
        want = dist_name.replace("-", "_").lower()
        for d in _md.distributions():
            n = (d.metadata["Name"] or "") if d.metadata else ""
            if n.replace("-", "_").lower() == want:
                return d.version
    except Exception:
        pass
    try:                                   # last resort: the module attribute
        m = __import__(module or dist_name)
        return getattr(m, "__version__", "?")
    except Exception:
        return None
versions = {}
for p in PKGS:
    v = pkg_version(p, mods[p])
    try:
        __import__(mods[p])
        installed = True
    except Exception:
        installed = False
    if installed and v:
        versions[p] = v
        note = ""
        if p == "pyspark" and v != EXPECTED["spark"]:
            note = f"(report says {EXPECTED['spark']})"
        if p == "vaderSentiment":
            note = "<-- must be IDENTICAL for D1 vs D2 comparability"
        if p == "zstandard":
            note = "(optional - only for .zst archives)"
        line(p, v, note)
    else:
        versions[p] = None
        if p == "zstandard":
            line(p, "not installed", "(fine - data/ is already .jsonl)")
        else:
            line(p, "MISSING", "<-- pip install it")

# ---------------------------------------------------------------- pip health
print("\n[pip launcher]")
scripts = os.path.join(sys.prefix, "Scripts" if platform.system() == "Windows" else "bin")
pip_exe = os.path.join(scripts, "pip.exe" if platform.system() == "Windows" else "pip")
if os.path.exists(pip_exe):
    try:
        r = subprocess.run([pip_exe, "--version"], capture_output=True, text=True, timeout=60)
        ok = r.returncode == 0 and os.path.abspath(sys.prefix) in r.stdout
        line("pip / pip.exe", "OK" if ok else "points at another path",
             "" if ok else "<-- broken by a rename/move; use 'python -m pip' instead")
        if not ok:
            print(f"    it says: {(r.stdout or r.stderr).strip()[:110]}")
    except Exception as e:
        line("pip / pip.exe", f"BROKEN ({type(e).__name__})",
             "<-- use 'python -m pip' instead")
else:
    line("pip / pip.exe", "not present", "use 'python -m pip'")

# ---------------------------------------------------------------- verdict
print("\n[verdict]")
problems = []
if not in_venv:
    problems.append("not inside a virtual environment")
if versions.get("pyspark") is None:
    problems.append("pyspark not importable")
if versions.get("vaderSentiment") is None:
    problems.append("vaderSentiment not importable - Table 6.2+ cannot be reproduced")
if platform.python_version() != EXPECTED["python"]:
    problems.append(f"python {platform.python_version()} != report's {EXPECTED['python']}")
if not versions.get("pandas"):
    problems.append("pandas not importable")
if versions.get("pyspark") and versions["pyspark"] != EXPECTED["spark"]:
    problems.append(f"pyspark {versions['pyspark']} != report's {EXPECTED['spark']}")
if problems:
    print("  WARN / ACTION:")
    for p in problems:
        print("   -", p)
    print("\n  Only *fix* these if Dataset 1 has NOT been run in this environment.")
    print("  If Dataset 1 already ran here, matching it matters more than matching the report;")
    print("  update Table 8.1 to the versions below instead.")
else:
    print("  all good - environment matches the report's Table 8.1")

print("\n[save this for the report + your own record]")
print("  python -m pip freeze > requirements_locked.txt")
print("  (keep that file; it is how you prove the run is reproducible)")
print("=" * 72)
