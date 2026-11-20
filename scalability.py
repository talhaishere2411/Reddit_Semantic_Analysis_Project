import subprocess, sys
path = sys.argv[1]
for frac in (0.1, 0.5, 1.0):
    for master in ("local[1]", "local[2]", "local[4]", "local[*]"):
        subprocess.run([sys.executable, "sentiment_pipeline.py", master, str(frac), path], check=True)
