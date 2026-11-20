"""Is the sentiment shift between periods REAL, or just a change in subreddit mix?

    python d2_sentiment_robustness.py

WHY THIS MATTERS
    Your results show LLM-era posts as clearly more positive than the 2019 control:
        control  avg_compound 0.0767, 47.49 % positive
        LLM era  avg_compound 0.1250, 54.11 % positive
    But the subreddit MIX changed a lot over the same window (from d2_prep_trim.py):

        Guitar          6.5 % of control  ->  12.9 % of LLM era
        tifu            6.5 %             ->   1.9 %
        TalesFromRetail 0.9 %             ->   0.1 %
        legaladvice    22.6 %             ->  17.5 %

    Guitar is one of Dataset 1's most POSITIVE subreddits (+0.231) and tifu one of the
    most NEGATIVE (-0.139). So part of the "sentiment rose" finding may be pure
    composition: the same communities, weighted differently.

    A reviewer who notices this can dismiss the conclusion. This script settles it with
    direct standardisation: apply the 2019 subreddit weights to the 2024-26 sentiment.

        standardised era mean = SUM over subreddits ( control_share_s * era_mean_s )

    If the standardised gap is much smaller than the raw gap, the shift is mostly
    composition and must be reported that way.

Outputs output_ai/posts_sentiment_by_subreddit.csv and prints the comparison.
"""
import os

import d2_paths as P
import pandas as pd
from pyspark.sql import functions as F

OUT = P.OUT_D2
SRC = os.path.join(OUT, "posts_scored.parquet")


def standardise(df):
    """Pure function so the maths is testable without Spark.

    df columns: subreddit, period, compound, label   (automated rows already removed)
    Returns dict with raw and standardised means.
    """
    g = (df.groupby(["subreddit", "period"])
           .agg(n=("compound", "size"),
                mean_c=("compound", "mean"),
                pos=("label", lambda s: (s == "positive").mean()),
                neg=("label", lambda s: (s == "negative").mean()))
           .reset_index())
    wide = g.pivot(index="subreddit", columns="period")
    if ("n", "control_pre_LLM") not in wide or ("n", "llm_era") not in wide:
        return None
    w = wide[("n", "control_pre_LLM")] / wide[("n", "control_pre_LLM")].sum()   # 2019 weights
    # restrict to subreddits present in both periods, then renormalise the weights
    both = w.index[(wide[("n", "llm_era")] > 0) & (wide[("n", "control_pre_LLM")] > 0)]
    w = w.loc[both] / w.loc[both].sum()

    ctrl_mean = (w * wide.loc[both, ("mean_c", "control_pre_LLM")]).sum()
    std_era = (w * wide.loc[both, ("mean_c", "llm_era")]).sum()
    raw_era = wide.loc[both, ("mean_c", "llm_era")].mul(
        wide.loc[both, ("n", "llm_era")]).sum() / wide.loc[both, ("n", "llm_era")].sum()

    ctrl_pos = (w * wide.loc[both, ("pos", "control_pre_LLM")]).sum()
    std_era_pos = (w * wide.loc[both, ("pos", "llm_era")]).sum()
    raw_era_pos = (wide.loc[both, ("pos", "llm_era")] * wide.loc[both, ("n", "llm_era")]).sum() \
        / wide.loc[both, ("n", "llm_era")].sum()

    return {
        "raw": {"control": ctrl_mean, "era": raw_era, "gap": raw_era - ctrl_mean,
                "control_pos": ctrl_pos, "era_pos": raw_era_pos,
                "gap_pos": raw_era_pos - ctrl_pos},
        "standardised": {"control": ctrl_mean, "era": std_era, "gap": std_era - ctrl_mean,
                         "era_pos": std_era_pos, "gap_pos": std_era_pos - ctrl_pos},
        "table": g, "weights": w,
    }


def _selftest():
    """Prove the standardisation works: build two periods where ONE subreddit differs in
    both sentiment and weight, and check that standardising removes the composition part."""
    rows = []
    # sub A: stable sentiment +0.2 ; sub B: stable sentiment -0.2
    # control: 50/50 ; era: 90/10  -> raw era mean rises even though nothing changed
    for sub, ctrl_n, era_n, val in (("A", 500, 900, 0.2), ("B", 500, 100, -0.2)):
        rows += [{"subreddit": sub, "period": "control_pre_LLM", "compound": val,
                  "label": "positive" if val > 0 else "negative"}] * ctrl_n
        rows += [{"subreddit": sub, "period": "llm_era", "compound": val,
                  "label": "positive" if val > 0 else "negative"}] * era_n
    r = standardise(pd.DataFrame(rows))
    raw, std = r["raw"]["gap"], r["standardised"]["gap"]
    print(f"  control mean        {r['raw']['control']:+.4f}")
    print(f"  era raw mean        {r['raw']['era']:+.4f}   gap {raw:+.4f}  <- looks like a rise")
    print(f"  era standardised    {r['standardised']['era']:+.4f}   gap {std:+.4f}  <- the truth")
    assert abs(raw - 0.16) < 1e-9, raw          # 0.9*0.2 + 0.1*(-0.2) = 0.16, vs 0.0
    assert abs(std) < 1e-9, std                 # standardising recovers 0.0
    print("  OK - standardisation removes a pure composition effect")


if __name__ == "__main__":
    if not os.path.isdir(SRC):
        raise SystemExit(f"[robust] missing {SRC} - run ai_comments_pipeline.py first")
    print("[robust] checking the maths on synthetic data first ...")
    _selftest()

    spark = P.spark("d2_robust")
    df = (spark.read.parquet(SRC)
              .filter(F.col("automated") == 0)
              .select("subreddit", "period", "compound", "label")
              .toPandas())                       # 2.6 M rows of 4 columns: safe
    print(f"\n[robust] rows: {len(df):,}")

    r = standardise(df)
    if r is None:
        raise SystemExit("[robust] need both periods present")

    P.ensure_dirs(OUT)
    r["table"].to_csv(os.path.join(OUT, "posts_sentiment_by_subreddit.csv"), index=False)

    raw, std = r["raw"], r["standardised"]
    print("\n" + "=" * 74)
    print("RAW vs STANDARDISED (2024-26 sentiment reweighted to the 2019 subreddit mix)")
    print("=" * 74)
    print(f"  avg compound")
    print(f"    control (2019)            {raw['control']:+.4f}")
    print(f"    LLM era, as observed      {raw['era']:+.4f}   gap {raw['gap']:+.4f}")
    print(f"    LLM era, standardised     {std['era']:+.4f}   gap {std['gap']:+.4f}")
    print(f"\n  positive share")
    print(f"    control (2019)            {raw['control_pos']*100:.2f} %")
    print(f"    LLM era, as observed      {raw['era_pos']*100:.2f} %   "
          f"gap {(raw['gap_pos'])*100:+.2f} pp")
    print(f"    LLM era, standardised     {std['era_pos']*100:.2f} %   "
          f"gap {(std['gap_pos'])*100:+.2f} pp")

    keep = 0 if raw["gap"] == 0 else std["gap"] / raw["gap"]
    print(f"\n  {keep*100:.0f} % of the observed compound gap survives the composition control")
    print("\n  HOW TO WORD IT IN THE REPORT")
    if abs(keep) < 0.5:
        print("    The difference is mostly a change in subreddit mix. Report: 'the apparent")
        print("    shift largely reflects the changed composition of the sample; after")
        print(f"    standardising to the 2019 mix the gap falls from {raw['gap']:+.4f} to "
              f"{std['gap']:+.4f}.'")
    else:
        print(f"    Most of the shift survives ({keep*100:.0f} %), so it is a real tone change")
        print("    and not only composition. Still state that the mix changed.")
    print("\n  per-subreddit table written to output_ai/posts_sentiment_by_subreddit.csv")
    print(r["table"].to_string(index=False)[:2000])
    spark.stop()
