import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
O = "output_ai/"

share = pd.read_csv(O + "comments_share_by_period.csv")
base = share.loc[share.period == "control_pre_LLM", "ai_like_pct"]
era = share.loc[share.period == "llm_era", "ai_like_pct"]
if len(base) and len(era):
    print("Control (pre-LLM) AI-like %:", float(base.iloc[0]), "| LLM-era AI-like %:", float(era.iloc[0]),
          "| excess over control (percentage points):", round(float(era.iloc[0] - base.iloc[0]), 3))

m = pd.read_csv(O + "comments_share_by_month.csv")
ax = m.plot(x="month", y="ai_like_pct", legend=False, title="AI-like comments per month (%)", rot=60)
ax.set_ylabel("% of comments"); plt.tight_layout(); plt.savefig(O + "fig_ai_share_monthly.png", dpi=150); plt.clf()

s = pd.read_csv(O + "comments_sentiment_ai_vs_human.csv")
s["group"] = s.period + " | " + s.ai_like.map({0: "human-like", 1: "AI-like"})
s.plot.bar(x="group", y="avg_compound", legend=False, title="Average VADER compound score", rot=30)
plt.ylabel("compound"); plt.tight_layout(); plt.savefig(O + "fig_sentiment_ai_vs_human.png", dpi=150); plt.clf()

s.set_index("group")[["pos_pct", "neu_pct", "neg_pct"]].plot.bar(stacked=True, title="Sentiment mix (%)", rot=30)
plt.tight_layout(); plt.savefig(O + "fig_sentiment_mix.png", dpi=150)
print(s[["period", "ai_like", "n", "avg_compound", "pos_pct", "neu_pct", "neg_pct"]])
