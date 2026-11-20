"""Prove (or disprove) that this environment reproduces Dataset 1's VADER labels.

Why: the venv used for Dataset 1 is gone. A pip cache tells you which wheels were
DOWNLOADED, not which versions were INSTALLED - so "the packages are cached" is not
evidence that the numbers will match. This settles it empirically instead.

It reads the Dataset 1 output you already have (output/labelled.parquet, which stores
the original `summary` text AND the `compound` score that was computed for it), recomputes
the score with the vaderSentiment installed RIGHT NOW, and compares.

    python check_vader_repro.py                       # 5000 rows, default
    python check_vader_repro.py --n 20000
    python check_vader_repro.py --parquet output/labelled.parquet

Verdict
    PASS      identical scores and labels -> Dataset 2 is directly comparable to
              Dataset 1. Record vaderSentiment's version in Table 8.1 and move on.
    DRIFT     scores moved slightly -> the lexicon changed. Either install the old
              version, or report that the comparison is approximate.
    FAIL      labels disagree -> do NOT compare Tables 6.2 and 6.7 as if one method.
"""
import argparse
import os
import sys

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default=os.path.join("output", "labelled.parquet"))
ap.add_argument("--n", type=int, default=5000)
a = ap.parse_args()

if not os.path.exists(a.parquet):
    raise SystemExit(f"[repro] not found: {a.parquet}\n"
                     f"        point --parquet at your Dataset 1 output "
                     f"(it must contain columns: summary, compound)")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from importlib import metadata as _md
    VADER_VER = _md.version("vaderSentiment")
except Exception:
    try:
        import vaderSentiment
        VADER_VER = getattr(vaderSentiment, "__version__", "unreported")
    except Exception:
        VADER_VER = "unknown"
import platform
print("=" * 72)
print("VADER REPRODUCIBILITY CHECK  (does this env reproduce Dataset 1?)")
print("=" * 72)
print(f"  python          {platform.python_version()}")
print(f"  vaderSentiment  {VADER_VER}")
print(f"  parquet         {a.parquet}")

df = pd.read_parquet(a.parquet)
need = {"summary", "compound"}
if not need.issubset(df.columns):
    raise SystemExit(f"[repro] {a.parquet} needs columns {sorted(need)}; has {sorted(df.columns)}")

df = df[["summary", "compound", "label"]].dropna() if "label" in df.columns else df[["summary", "compound"]].dropna()
if len(df) > a.n:
    df = df.sample(a.n, random_state=42)
print(f"  rows compared   {len(df):,}")

an = SentimentIntensityAnalyzer()
new = df["summary"].astype(str).map(lambda t: an.polarity_scores(t)["compound"])
old = df["compound"].astype(float)

diff = (new - old).abs()
exact = int((diff <= 1e-9).sum())
n = len(df)


def lab(c):
    return "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")


new_lab = new.map(lab)
print("\n--- score agreement ---")
print(f"  identical scores (<=1e-9) : {exact:,} / {n:,}  ({100*exact/n:.4f} %)")
print(f"  max |difference|          : {diff.max():.6g}")
print(f"  mean |difference|         : {diff.mean():.6g}")

if "label" in df.columns:
    agree = (new_lab.values == df["label"].values).sum()
    print("\n--- label agreement (thresholds +-0.05) ---")
    print(f"  same label                : {agree:,} / {n:,}  ({100*agree/n:.4f} %)")
    flips = df.assign(new=new_lab).loc[df["label"].values != new_lab.values]
    if len(flips):
        print("  sample of flipped labels:")
        print(flips.head(5).to_string(index=False)[:1200])
        piv = pd.crosstab(df["label"], new_lab)
        print("\n  old label (rows) vs new label (columns):")
        print(piv.to_string())

print("\n--- verdict ---")
label_agree_pct = (100 * agree / n) if "label" in df.columns else 100.0

if exact == n:
    print("  PASS - this environment reproduces Dataset 1 exactly.")
    print("         Dataset 2 sentiment is directly comparable. Suitable sentence for the report:")
    print(f"         'VADER labelling was re-verified against the Dataset 1 output on {n:,}")
    print("          records; all scores were identical.'")
    print(f"         Record vaderSentiment {VADER_VER} in Table 8.1.")
elif label_agree_pct >= 99.9:
    print("  DRIFT (labels stable) - scores moved, but less than 0.1% of labels changed.")
    print("         Table 6.2 / 6.5 / 6.6 / 7.4 are label-based and stay valid.")
    print("         Table 6.7 (avg compound) may shift in the 3rd decimal: re-check it and")
    print("         mention in Section 6.7 that VADER was re-installed between the two runs.")
    print("         Cleanest fix: pip install the original vaderSentiment version and re-run")
    print("         this check until it says PASS.")
elif label_agree_pct >= 99.0:
    print("  DRIFT (labels affected) - a small share of labels changed.")
    print("         Report the comparison as approximate and state the disagreement rate,")
    print("         or install the Dataset 1 vaderSentiment version and redo Dataset 2.")
else:
    print("  FAIL - this environment does NOT reproduce Dataset 1.")
    print("         Do not present Tables 6.2 and 6.7 as one consistent method until the")
    print("         original vaderSentiment version is installed.")
print("=" * 72)
