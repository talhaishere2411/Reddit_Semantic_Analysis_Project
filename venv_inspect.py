"""Inspect a .venv copied in from another project folder.

    python venv_inspect.py                  # looks for .venv / venv in the current folder
    python venv_inspect.py path\\to\\.venv

A venv is NOT relocatable: absolute paths are baked into pyvenv.cfg, into the
activate scripts, and into the pip launcher. Moving the folder breaks some of it
and silently keeps the rest working. This prints which is which on YOUR machine,
and - most usefully - recovers the exact package versions straight from
site-packages/*.dist-info, which needs no working Python, no pip and no activation.

That version list is what you lost when the old venv was retired; it is also what
Table 8.1 of the report needs.
"""
import glob
import json
import os
import platform
import re
import subprocess
import sys

VENV = sys.argv[1] if len(sys.argv) > 1 else next(
    (c for c in (".venv", "venv", "env", ".env") if os.path.isdir(c)), None)
if not VENV or not os.path.isdir(VENV):
    raise SystemExit("[inspect] no venv found — pass the path explicitly")

IS_WIN = platform.system() == "Windows"
PY = os.path.abspath(os.path.join(VENV, "Scripts", "python.exe") if IS_WIN
                     else os.path.join(VENV, "bin", "python"))

KEY = ["pyspark", "vaderSentiment", "pandas", "numpy", "pyarrow",
       "matplotlib", "zstandard", "py4j"]
REPORT = {"pyspark": "4.3.0", "python": "3.13.4"}


def hdr(t):
    print("\n" + "-" * 70 + f"\n{t}\n" + "-" * 70)


# ---------------------------------------------------------------- 1. cfg
hdr("1. pyvenv.cfg (read without running anything)")
cfg = {}
cfg_path = os.path.join(VENV, "pyvenv.cfg")
if os.path.exists(cfg_path):
    for ln in open(cfg_path, encoding="utf8", errors="replace"):
        if "=" in ln:
            k, v = ln.split("=", 1)
            cfg[k.strip()] = v.strip()
    for k in ("home", "version", "executable", "command", "include-system-site-packages"):
        if k in cfg:
            print(f"  {k:<28} {cfg[k]}")
else:
    print("  !! pyvenv.cfg missing — this is not a venv")

base_ok = None
if cfg.get("home"):
    home = cfg["home"]
    base_ok = os.path.isdir(home) or os.path.exists(os.path.join(home, "python.exe"))
    print(f"\n  base interpreter present?    {'YES — venv is revivable' if base_ok else 'NO — this venv is DEAD, rebuild it'}")
    print(f"  (looked for {home})")

# ---------------------------------------------------------------- 2. versions
hdr("2. package versions, read from *.dist-info (needs no working pip/python)")
sp = glob.glob(os.path.join(VENV, "Lib", "site-packages")) + \
     glob.glob(os.path.join(VENV, "lib", "python*", "site-packages"))
versions, all_pkgs = {}, {}
for d in sp:
    for di in glob.glob(os.path.join(d, "*.dist-info")):
        name = os.path.basename(di).rsplit("-", 1)[0]
        ver = os.path.basename(di)[:-len(".dist-info")].rsplit("-", 1)[-1]
        m = re.match(r"(.+)-([0-9][^-]*)$", os.path.basename(di)[:-10])
        if m:
            name, ver = m.group(1), m.group(2)
        all_pkgs[name] = ver
if not all_pkgs:
    print("  no dist-info found — site-packages empty or the path is unusual")
for k in KEY:
    hit = next((v for n, v in all_pkgs.items() if n.lower() == k.lower()), None)
    if hit:
        versions[k] = hit
        flag = ""
        if k in REPORT and hit != REPORT[k]:
            flag = f"   (report Table 8.1 says {REPORT[k]})"
        if k == "pyspark" and hit == REPORT["pyspark"]:
            flag = "   matches the report"
        print(f"  {k:<18} {hit}{flag}")
    else:
        print(f"  {k:<18} -- not installed --")

if all_pkgs:
    print(f"\n  {len(all_pkgs)} packages total in site-packages")
    print(f"  python in this venv        {cfg.get('version', '?')}"
          + ("" if cfg.get("version", "").startswith(REPORT["python"].rsplit(".", 1)[0])
             else "   (report says 3.13.4)"))

# ---------------------------------------------------------------- 3. does it run
hdr("3. does the interpreter still run from this location?")
ok_run = False
if os.path.exists(PY):
    r = subprocess.run([PY, "-c",
                        "import sys;print(sys.prefix);print(sys.base_prefix);"
                        "print('.'.join(map(str,sys.version_info[:3])))"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        ok_run = True
        pref, base, ver = (r.stdout.strip().splitlines() + ["", "", ""])[:3]
        print(f"  runs            YES   ({PY})")
        print(f"  sys.prefix      {pref}")
        print(f"  sys.base_prefix {base}")
        print(f"  python version  {ver}")
        good = os.path.abspath(pref) == os.path.abspath(VENV)
        print(f"  resolves to THIS folder? {'YES — imports will work' if good else 'NO — points elsewhere'}")
    else:
        print(f"  runs            NO    {r.stderr.strip()[:200]}")
else:
    print(f"  !! no interpreter at {PY}")

# ---------------------------------------------------------------- 4. pip / activate
hdr("4. what is broken: pip launcher and activate")
if IS_WIN:
    launcher = os.path.join(VENV, "Scripts", "pip.exe")
    act = os.path.join(VENV, "Scripts", "activate.bat")
else:
    launcher = os.path.join(VENV, "bin", "pip")
    act = os.path.join(VENV, "bin", "activate")

if os.path.exists(launcher):
    launcher = os.path.abspath(launcher)
    try:
        r = subprocess.run([launcher, "--version"], capture_output=True, text=True)
        if r.returncode == 0:
            line = r.stdout.strip()
            stale = os.path.abspath(VENV) not in line
            print(f"  pip             runs, but reports: {line[:110]}")
            if stale:
                print("                  ^^ it is using the OLD path — silently the WRONG env!")
        else:
            print("  pip             BROKEN:", (r.stderr or r.stdout).strip()[:140])
    except (OSError, FileNotFoundError) as e:
        # the launcher's recorded interpreter no longer exists
        print("  pip             BROKEN — launcher cannot find its interpreter")
        print(f"                  ({type(e).__name__}: {e})")
        print("                  Windows shows this as: Fatal error in launcher: Unable")
        print("                  to create process using C:\\...old...\\python.exe")
else:
    print("  pip launcher    not present")
if ok_run:
    r = subprocess.run([PY, "-m", "pip", "--version"], capture_output=True, text=True)
    print(f"  python -m pip   {'works: ' + r.stdout.strip()[:100] if r.returncode == 0 else 'BROKEN'}")

if os.path.exists(act):
    txt = open(act, encoding="utf8", errors="replace").read()
    m = re.search(r"VIRTUAL_ENV[=:]\s*\"?([^\"\n]*(?:\.venv|venv)[^\"\n]*)", txt)
    if m:
        printed = m.group(1).strip('$()" ')
        stale = os.path.abspath(VENV) not in printed
        print(f"  activate sets   VIRTUAL_ENV -> {printed}")
        print(f"                  {'STALE — activation would put the OLD folder on PATH' if stale else 'correct'}")

# ---------------------------------------------------------------- 5. imports
hdr("5. can the key packages actually be imported?")
if ok_run:
    code = ("import importlib,sys\n"
            "for m in ['pyspark','vaderSentiment','pandas','pyarrow','matplotlib','zstandard']:\n"
            "    try:\n"
            "        mod=importlib.import_module(m); print(m,'OK',getattr(mod,'__version__','?'))\n"
            "    except Exception as e: print(m,'FAIL',type(e).__name__)\n")
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    for ln in (r.stdout or r.stderr).strip().splitlines():
        print("  " + ln)

# ---------------------------------------------------------------- 6. verdict
hdr("6. verdict and what to do")
usable = ok_run and base_ok
if usable:
    print("  The environment WORKS if you never activate it and never call the pip launcher:")
    print(f"      {os.path.relpath(PY)} run_d2.py")
    print(f"      {os.path.relpath(PY)} -m pip install <pkg>")
    print("  Do NOT use `activate`, and do NOT use `pip` / `pip.exe` directly.")
else:
    print("  This venv is NOT usable — rebuild it (see below).")

if versions:
    print("\n  Recovered versions -> save as requirements_locked.txt (this is your Table 8.1):")
    for k in KEY:
        if k in versions:
            print(f"      {k}=={versions[k]}")

print("\n  RECOMMENDED (2 minutes, removes all ambiguity):")
print("      python -m venv .venv_new")
if IS_WIN:
    print("      .venv_new\\Scripts\\python.exe -m pip install --upgrade pip")
    print("      .venv_new\\Scripts\\python.exe -m pip install "
          + " ".join(f"{k}=={v}" for k, v in versions.items() if k in KEY))
else:
    print("      .venv_new/bin/python -m pip install --upgrade pip")
    print("      .venv_new/bin/python -m pip install "
          + " ".join(f"{k}=={v}" for k, v in versions.items() if k in KEY))
print("      # verify it reproduces Dataset 1 before trusting it:")
print("      .venv_new/<python> check_vader_repro.py")
print("\n  Keeping the old .venv as-is is only acceptable if you always call its")
print("  python.exe by full path. It is fragile, and it must not be submitted with the")
print("  project (a pyspark venv is several hundred MB).")
print("=" * 70)
