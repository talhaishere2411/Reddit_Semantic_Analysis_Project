"""Print every yellow [ ] cell in the report, in report order, from output_ai/*.csv.

    python d2_fill_report_tables.py

Nothing here invents a number: a missing source file prints MISSING instead of a
plausible value. Run after the pipeline (+ the validation step).
"""
import json
import os

import d2_paths as P
import pandas as pd

KIND = "posts"
O = P.OUT_D2


def rd(name):
    p = os.path.join(O, f"{KIND}_{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def summary():
    p = os.path.join(O, f"{KIND}_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------- 6.1 / 4.2
hdr("TABLE 6.1  rows before/after cleaning (posts only — comments column deleted)")
s = summary()
if not s:
    print("MISSING output_ai/posts_summary.json")
else:
    print(f"raw rows read      : {s['rows_raw']:,}")
    print(f"after cleaning     : {s['rows_clean']:,}")
    print(f"retained           : {s['retained_pct']} %")
    print(f"fraction           : {s['fraction']}   <- row label: 'Dataset 2 = {s['fraction']}'")
    print(f"stage seconds      : {s.get('stage_seconds')}")
print("§4.2 'Size / records': fill file count + GB from `du -sh data/` and the post count")
print("      from d2_prep_trim.py's 'rows kept' line.")

# ---------------------------------------------------------------- 6.5
hdr("TABLE 6.5  AI-like share by period   +   §6.5 excess over control")
d = rd("share_by_period")
excess = None
if d is None:
    print("MISSING posts_share_by_period.csv")
else:
    order = {"control_pre_LLM": 0, "transition": 1, "llm_era": 2}
    d = d.sort_values("period", key=lambda c: c.map(order))
    print(d[["period", "n", "ai_like_pct", "accusation_pct", "automated_pct"]]
          .rename(columns={"n": "rows", "ai_like_pct": "AI-like %",
                           "accusation_pct": "accusations %",
                           "automated_pct": "automated %"}).to_string(index=False))
    g = d.set_index("period").ai_like_pct
    if "control_pre_LLM" in g and "llm_era" in g:
        excess = round(float(g["llm_era"] - g["control_pre_LLM"]), 3)
        print(f"\nEXCESS OVER CONTROL = {excess} percentage points")
        print("Write it as: 'AI-like share rose from C% in the control period to E% in the")
        print("LLM era, an excess of X percentage points.' Never 'X% of posts are AI'.")
    else:
        print("!! no control or llm_era row -> check the 2019 files before writing this")

# ---------------------------------------------------------------- 6.6
hdr("TABLE 6.6  subreddits with highest and lowest AI-like share")
d = rd("share_by_subreddit")
if d is None:
    print("MISSING posts_share_by_subreddit.csv")
else:
    w = d.pivot_table(index="subreddit", columns="period", values="ai_like_pct", aggfunc="first")
    n = d.pivot_table(index="subreddit", columns="period", values="n", aggfunc="first")
    if {"control_pre_LLM", "llm_era"} <= set(w.columns):
        w = w[["control_pre_LLM", "llm_era"]].rename(
            columns={"control_pre_LLM": "control AI-like %", "llm_era": "LLM era AI-like %"})
        w["difference (pp)"] = (w["LLM era AI-like %"] - w["control AI-like %"]).round(3)
        print(w.sort_values("difference (pp)", ascending=False).to_string())
        print("\nrow counts (be honest about small cells):")
        print(n.to_string())
        print("\nSuggested 6.6 rows = top 3 and bottom 3 by difference:")
        sw = w.sort_values("difference (pp)", ascending=False)
        print(pd.concat([sw.head(3), sw.tail(3)]).to_string())
        missing = 10 - len(w)
        if missing:
            print(f"\n!! only {len(w)} subreddits qualified: {missing} had <200 rows in a period")
            print("   (the HAVING COUNT(*)>=200 filter). Mention it or lower the threshold.")
    else:
        print("!! missing a period in the pivot; raw table:"); print(w.to_string())

# ---------------------------------------------------------------- 6.7 / 6.8
hdr("TABLE 6.7  sentiment of AI-like vs human-like (automated excluded)")
d = rd("sentiment_ai_vs_human")
print(d.to_string(index=False) if d is not None else "MISSING")
print("Report rows: Control/human-like, LLM era/human-like, LLM era/AI-like.")
print("CAVEAT: avg_upvotes is not comparable 2019 vs 2023-26 (score display changed,")
print("        recent rows unsettled). Footnote it or drop the column.")

hdr("TABLE 6.8  sentiment of accusation posts (LLM era)")
d = rd("sentiment_accusations")
if d is None:
    print("MISSING")
else:
    print(d[d.period == "llm_era"].to_string(index=False))

# ---------------------------------------------------------------- 6.10
hdr("TABLE 6.10  cross-dataset row (Overall sentiment, AI-like share)")
d = rd("sentiment_ai_vs_human")
if d is not None:
    t = d[d.period == "llm_era"]
    if len(t):
        n = t.n.sum()
        print(f"Dataset 2 posts, LLM era overall: pos {round((t.pos_pct*t.n).sum()/n,2)}%, "
              f"neu {round((t.neu_pct*t.n).sum()/n,2)}%, neg {round((t.neg_pct*t.n).sum()/n,2)}% "
              f"(n={int(n):,})")
print("'AI-like share' cell = the Table 6.5 llm_era row.")

# ---------------------------------------------------------------- 7.3
hdr("TABLE 7.3  Dataset 2 runtime and throughput")
if s:
    tot = sum(s.get("stage_seconds", {}).values())
    print(f"rows cleaned : {s['rows_clean']:,}")
    print(f"total seconds: {round(tot,1)}")
    print(f"throughput   : {round(s['rows_clean']/tot,1) if tot else 0} rows/s")
    print(f"cores        : {s.get('master')}")
    print(f"stages       : {s.get('stage_seconds')}")
    print("Add one line: the json->parquet trim time is separate (not in this total).")
else:
    print("MISSING")

# ---------------------------------------------------------------- 7.4
hdr("TABLE 7.4  validation of the AI-like method")
d = rd("share_by_period")
if d is not None:
    g = d.set_index("period").ai_like_pct
    if {"control_pre_LLM", "llm_era"} <= set(g.index):
        print(f"control {g['control_pre_LLM']:.3f}%  |  LLM era {g['llm_era']:.3f}%  |  "
              f"excess {round(float(g['llm_era']-g['control_pre_LLM']),3)} pp")
print("Then run:  python validate_sample.py score")
print("           (needs the labelled output_ai/validation_sample.csv)")

# ---------------------------------------------------------------- figures
hdr("FIGURES 6.3 / 6.4 / 6.5")
print("Run:  python d2_charts.py")
for f, fig in (("ai_share_monthly", "6.3"), ("sentiment_ai_vs_human", "6.4"),
               ("sentiment_mix", "6.5")):
    p = os.path.join(O, f"fig_{KIND}_{f}.png")
    print(f"  {'OK  ' if os.path.exists(p) else 'TODO'} {p}  -> Figure {fig}")

# ---------------------------------------------------------------- sensitivity
hdr("TABLE 7.1 last row — threshold sensitivity")
p = os.path.join(O, f"{KIND}_threshold_sensitivity.csv")
print(pd.read_csv(p).to_string(index=False) if os.path.exists(p)
      else "Run:  python d2_threshold_sensitivity.py")

print("\nStill to write by hand: §6.5 interpretation, §6.6 interpretation,")
print("§7 'Observations (Dataset 2 and validation)'. Template: REPORT_EDITS_POSTS_ONLY.md A16.")
