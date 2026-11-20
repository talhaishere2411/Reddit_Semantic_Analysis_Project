"""Ask the live Spark UI whether the job is actually progressing.

    python d2_spark_progress.py              # one snapshot
    python d2_spark_progress.py --watch      # poll every 5 s, show deltas
    python d2_spark_progress.py --watch 10   # poll every 10 s

Run it in a SECOND terminal while run_d2.py is working. It reads Spark's REST API on
localhost:4040, so it needs no Spark session of its own and cannot disturb the running job.

What the columns mean for THIS pipeline:
  active tasks   > 0                          -> work is happening right now
  completed      rising between polls         -> tasks are FINISHING (the real signal)
  shuffle r/w    near zero on the VADER stage -> expected (it is map-only, CPU-bound)
  shuffle r/w    large on the aggregation stage -> expected (SQL GROUP BY)
  duration       climbing, active 0 for long  -> one long task; still fine, but see below
"""
import json
import sys
import time
import urllib.request

BASE = "http://localhost:4040/api/v1"
WATCH = "--watch" in sys.argv
INTERVAL = 5
for a in sys.argv[1:]:
    if a.isdigit():
        INTERVAL = int(a)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=5) as r:
        return json.load(r)


def snapshot():
    apps = get("/applications")
    if not apps:
        return None
    app = apps[0]
    aid = app["id"]
    stages = get(f"/applications/{aid}/stages?status=active")
    if not stages:
        stages = get(f"/applications/{aid}/stages?status=pending")
    execs = get(f"/applications/{aid}/executors")
    return app, stages, execs


def gb(n):
    return f"{n/1e9:.2f} GB" if n else "-"


def show(prev=None):
    try:
        snap = snapshot()
    except urllib.error.URLError as e:
        print(f"[ui] cannot reach {BASE}  ({e.reason})")
        print("     Spark publishes this only while an app is running. Start run_d2.py first.")
        return None
    if snap is None:
        print("[ui] no application registered yet")
        return None
    app, stages, execs = snap
    print("=" * 78)
    print(f"  app: {app.get('name')}   id: {app.get('id')}   "
          f"uptime: {round((time.time()*1000 - app.get('startTime', 0))/1000)} s")
    if not stages:
        print("  no ACTIVE stage -> between stages (planning, or writing, or finished)")
    for s in stages:
        done, active, total = (s.get("numCompleteTasks", 0), s.get("numActiveTasks", 0),
                               s.get("numTasks", 0))
        sr = s.get("shuffleReadBytes", 0)
        sw = s.get("shuffleWriteBytes", 0)
        print(f"\n  stage {s.get('stageId')} (attempt {s.get('attemptId')}) "
              f"{s.get('name', '')[:52]}")
        print(f"    tasks     {done}/{total} complete, {active} active")
        print(f"    shuffle   read {gb(sr)}   write {gb(sw)}")
        print(f"    metrics   input {gb(s.get('inputBytes',0))}  "
              f"output {gb(s.get('outputBytes',0))}")
        if prev and s.get("stageId") in prev:
            d = done - prev[s["stageId"]]
            rate = d / INTERVAL if INTERVAL else 0
            eta = f" | ~{round((total-done)/rate)} s left" if rate > 0 else ""
            print(f"    delta     +{d} tasks in {INTERVAL}s ({rate:.2f}/s){eta}")
        if active == 0 and done == total and total:
            print("    -> stage finished")
        elif active:
            print("    -> WORKING (CPU-bound is normal here; fans are expected)")
        else:
            print("    -> no active tasks but not complete: consecutive task failures")
            print("       or a very long first task. Check the UI's Tasks table.")
    print("\n  executors (CPU time is the ground truth that work is happening):")
    tot_cpu = 0
    for e in execs:
        if e.get("id") == "driver" or e.get("isActive"):
            cpu = e.get("totalCpuTime", 0) or 0
            tot_cpu += cpu
            print(f"    id={e.get('id'):>3}  tasks={e.get('totalTasks',0):>6}  "
                  f"cpu={cpu/1e9:8.1f} cpu-s  mem={gb(e.get('memoryUsed',0))}")
    if prev and "__cpu__" in prev:
        d = (tot_cpu - prev["__cpu__"]) / 1e9
        print(f"    delta     +{d:.1f} cpu-seconds in {INTERVAL}s "
              f"-> {'PROGRESSING' if d > 1 else 'IDLE or swapping'}")
    out = {s.get("stageId"): s.get("numCompleteTasks", 0) for s in stages}
    out["__cpu__"] = tot_cpu
    return out


if not WATCH:
    show()
    print("\n(run with --watch to poll and show deltas)")
else:
    prev = None
    try:
        while True:
            prev = show(prev) or prev
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n[ui] stopped")
