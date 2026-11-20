"""The three Dataset 2 figures, posts-only, straight into output_ai/.
Standalone: you do NOT need to patch your ai_charts.py (which is hard-coded to
comments_* and would produce nothing on a posts-only run).

    python d2_charts.py [posts]

Output (all created under output_ai/):
    fig_posts_ai_share_monthly.png        -> Figure 6.3
    fig_posts_sentiment_ai_vs_human.png   -> Figure 6.4
    fig_posts_sentiment_mix.png           -> Figure 6.5
"""
import os
import sys

import d2_paths as P
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KIND = sys.argv[1] if len(sys.argv) > 1 else "posts"
O = P.OUT_D2
P.ensure_dirs(O)


def rd(name):
    p = os.path.join(O, f"{KIND}_{name}.csv")
    if not os.path.exists(p):
        raise SystemExit(f"[charts] missing {p} — run ai_comments_pipeline.py first")
    return pd.read_csv(p)


# ---------------------------------------------------------------- Figure 6.3
m = rd("share_by_month").sort_values("month")
# split on the period boundary, NOT on a hard-coded year, so the titles always
# match the data actually retrieved (here: control 2019, LLM era 2024-2026)
ctrl = m[m.month < "2022-12"]
era = m[m.month >= "2022-12"]


def span(d, fallback):
    return f"{d.month.min()} to {d.month.max()}" if len(d) else fallback


fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
for a, d, t in ((ax[0], ctrl, f"control: {span(ctrl, 'none')}"),
                (ax[1], era, f"LLM era: {span(era, 'none')}")):
    if len(d):
        a.plot(d.month, d.ai_like_pct, marker="o", ms=3)
    a.set_title(t)
    a.tick_params(axis="x", rotation=90)
    a.grid(alpha=.3)
ax[0].set_ylabel("% AI-like")
fig.suptitle(f"AI-like {KIND} per month (%)  — two panels: the control period and the "
             f"LLM era are not contiguous")
plt.tight_layout()
plt.savefig(f"{O}/fig_{KIND}_ai_share_monthly.png", dpi=150)
plt.clf()

# ---------------------------------------------------------------- Figure 6.4
s = rd("sentiment_ai_vs_human").copy()
s["group"] = s.period + " | " + s.ai_like.map({0: "human-like", 1: "AI-like"})
# do not plot a group with n < 100: it invites a reviewer to call the chart misleading
ok = s[s.n >= 100]
if len(ok) < len(s):
    print("[charts] low-n groups excluded from Figures 6.4/6.5:",
          s[s.n < 100][["period", "ai_like", "n"]].to_dict("records"))
ok.plot.bar(x="group", y="avg_compound", legend=False,
            title=f"Average VADER compound score, {KIND} (automated accounts excluded)")
plt.ylabel("compound")
plt.axhline(0, color="k", lw=.8)
plt.tight_layout()
plt.savefig(f"{O}/fig_{KIND}_sentiment_ai_vs_human.png", dpi=150)
plt.clf()

# ---------------------------------------------------------------- Figure 6.5
ok.set_index("group")[["pos_pct", "neu_pct", "neg_pct"]].plot.bar(
    stacked=True, title=f"Sentiment mix (%) of {KIND}")
plt.tight_layout()
plt.savefig(f"{O}/fig_{KIND}_sentiment_mix.png", dpi=150)
plt.clf()

# ---------------------------------------------------------------- excess
share = rd("share_by_period").set_index("period").ai_like_pct
if "control_pre_LLM" in share and "llm_era" in share:
    c, e = float(share["control_pre_LLM"]), float(share["llm_era"])
    print(f"[charts] control {c:.3f}%  ->  LLM era {e:.3f}%   "
          f"excess = {round(e - c, 3)} percentage points")
    print(f"[charts] that excess is the §6.5 sentence and the Table 7.4 row.")

print(f"[charts] wrote 3 figures to {O}/")
print(s[["period", "ai_like", "n", "avg_compound", "pos_pct", "neu_pct", "neg_pct"]]
      .to_string(index=False))
