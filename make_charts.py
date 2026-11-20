import glob, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("output/dist.csv")
d.plot.bar(x="label", y="n", legend=False, title="Sentiment distribution", rot=0)
plt.ylabel("posts"); plt.tight_layout(); plt.savefig("output/fig_dist.png", dpi=150); plt.clf()

s = pd.read_csv("output/subreddit.csv")
if len(s) >= 2:
    k = min(10, len(s) // 2)
    pd.concat([s.head(k), s.tail(k)]).plot.barh(x="subreddit", y="avg_compound", legend=False,
        title="Most positive / most negative subreddits")
    plt.tight_layout(); plt.savefig("output/fig_subreddits.png", dpi=150); plt.clf()

rows = []
for f in glob.glob("output/summary_*.json"):
    j = json.load(open(f))
    t = sum(j["stage_seconds"].values()) + sum(m["train_seconds"] for m in j["ml"])
    rows.append({"master": j["master"], "fraction": j["fraction"], "total_s": round(t, 1)})
r = pd.DataFrame(rows).pivot(index="master", columns="fraction", values="total_s")
r.to_csv("output/scalability.csv"); print(r)
r.T.plot(marker="o", title="Total runtime vs data fraction")
plt.ylabel("seconds"); plt.tight_layout(); plt.savefig("output/fig_runtime.png", dpi=150)
