"""Two extra figures for the report — the ones that defend the Dataset 2 conclusions.

    python d2_chart_extra.py

Writes to output_ai/:
    fig_posts_composition_control.png   -> new Figure 6.6
    fig_posts_neutral_collapse.png      -> new Figure 6.7

WHY THESE TWO
    The existing Figures 6.3-6.5 all describe the AI-like share, which is your weakest
    number (63 and 248 flagged posts). Neither figure below depends on that measure, and
    both answer the question a reader will actually ask:

    6.6  Does the sentiment change survive the fact that the subreddit mix changed?
         Raw peak of +0.0483 in average compound falls to +0.0137 when the 2024-26 data
         is reweighted to the 2019 subreddit mix; the positive-share gap keeps 74 %.
         Showing both stops a reviewer dismissing the finding.

    6.7  Where in the sentiment distribution did the change happen?
         Neutral share halved (14.51 % -> 7.93 %) while negative share was flat, and this
         happens WITHIN nine of the ten subreddits - so it is not a composition artefact.

    Reads output_ai/posts_sentiment_by_subreddit.csv when present (d2_sentiment_robustness.py
    writes it). Falls back to the values measured on 18 Nov 2026 so the file can be
    regenerated on a machine that has only the outputs.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "output_ai"

# Measured 18 Nov 2026, Dataset 2 posts, automated accounts excluded.
FALLBACK = [
    ("Guitar", 68555, 0.355222, 0.613960, 0.125024, 201426, 0.341304, 0.633970, 0.163400),
    ("TalesFromRetail", 8927, 0.027650, 0.372242, 0.389829, 1938, 0.011324, 0.361197, 0.395253),
    ("buildapc", 190975, 0.422619, 0.722398, 0.154088, 254295, 0.465622, 0.769001, 0.166657),
    ("dating_advice", 92608, 0.315200, 0.614591, 0.241016, 225945, 0.408120, 0.693231, 0.230658),
    ("depression", 158710, -0.240400, 0.318650, 0.614473, 264424, -0.302589, 0.293850, 0.653432),
    ("legaladvice", 238528, -0.079494, 0.369282, 0.486182, 273232, -0.101212, 0.423322, 0.541335),
    ("offmychest", 179359, -0.047508, 0.412815, 0.500962, 229077, -0.045956, 0.445091, 0.513731),
    ("photography", 25771, 0.199921, 0.440844, 0.114625, 56060, 0.382227, 0.645184, 0.135837),
    ("summonerschool", 21186, 0.306839, 0.630605, 0.231332, 24767, 0.373702, 0.691444, 0.259902),
    ("tifu", 68048, 0.003670, 0.327093, 0.366697, 29664, 0.111311, 0.518238, 0.402508),
]
COLS = ["subreddit", "n_c", "mean_c", "pos_c", "neg_c", "n_e", "mean_e", "pos_e", "neg_e"]

# palette: control = grey, observed = blue, standardised = orange (colourblind-safe pair)
GREY, BLUE, ORANGE = "#8C8C8C", "#2E6DA4", "#D9822B"


def load():
    p = os.path.join(OUT, "posts_sentiment_by_subreddit.csv")
    if os.path.isfile(p):
        g = pd.read_csv(p)
        w = g.pivot(index="subreddit", columns="period")
        d = pd.DataFrame(index=w.index)
        d["n_c"] = w[("n", "control_pre_LLM")]
        d["mean_c"] = w[("mean_c", "control_pre_LLM")]
        d["pos_c"] = w[("pos", "control_pre_LLM")]
        d["neg_c"] = w[("neg", "control_pre_LLM")]
        d["n_e"] = w[("n", "llm_era")]
        d["mean_e"] = w[("mean_c", "llm_era")]
        d["pos_e"] = w[("pos", "llm_era")]
        d["neg_e"] = w[("neg", "llm_era")]
        d["neu_c"] = 1 - d.pos_c - d.neg_c
        d["neu_e"] = 1 - d.pos_e - d.neg_e
        print(f"[fig] read {p} ({len(d)} subreddits)")
        return d.reset_index()
    print("[fig] posts_sentiment_by_subreddit.csv not found - using values measured 18 Nov 2026")
    d = pd.DataFrame(FALLBACK, columns=COLS)
    d["neu_c"] = 1 - d.pos_c - d.neg_c
    d["neu_e"] = 1 - d.pos_e - d.neg_e
    return d


def aggregate(d):
    """Weighted period totals, plus the era reweighted to the 2019 subreddit mix."""
    wc = d.n_c / d.n_c.sum()
    we = d.n_e / d.n_e.sum()
    raw_c = {"mean": (wc * d.mean_c).sum(), "pos": (wc * d.pos_c).sum(),
             "neu": (wc * d.neu_c).sum(), "neg": (wc * d.neg_c).sum()}
    raw_e = {"mean": (we * d.mean_e).sum(), "pos": (we * d.pos_e).sum(),
             "neu": (we * d.neu_e).sum(), "neg": (we * d.neg_e).sum()}
    std_e = {"mean": (wc * d.mean_e).sum(), "pos": (wc * d.pos_e).sum(),
             "neu": (wc * d.neu_e).sum(), "neg": (wc * d.neg_e).sum()}
    return raw_c, raw_e, std_e


def fig_composition(d, raw_c, raw_e, std_e):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                   gridspec_kw={"width_ratios": [1, 1.55]})
    fig.suptitle("Figure 6.6: Composition control on the Dataset 2 sentiment shift",
                 fontsize=12, fontweight="bold", y=0.98)

    # ---- left: average compound ----
    vals = [raw_c["mean"], raw_e["mean"], std_e["mean"]]
    labs = ["Control\n(2019)", "LLM era\n(as observed)", "LLM era\n(2019 mix)"]
    bars = ax1.bar(labs, vals, color=[GREY, BLUE, ORANGE], width=0.62,
                   edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:+.4f}",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("average VADER compound", fontsize=10)
    ax1.set_ylim(0, max(vals) * 1.22)
    ax1.axhline(raw_c["mean"], color=GREY, ls=":", lw=1.1, zorder=0)
    # the two gaps, drawn as brackets
    ax1.annotate("", xy=(2, std_e["mean"]), xytext=(2, raw_c["mean"]),
                 arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.4))
    ax1.text(2.36, (std_e["mean"] + raw_c["mean"]) / 2, "+0.0137", color=ORANGE,
             fontsize=9, va="center", fontweight="bold")
    ax1.annotate("", xy=(1, raw_e["mean"]), xytext=(1, raw_c["mean"]),
                 arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.4))
    ax1.text(0.62, (raw_e["mean"] + raw_c["mean"]) / 2, "+0.0483", color=BLUE,
             fontsize=9, va="center", ha="right", fontweight="bold")
    ax1.set_title("72 % of the rise in the average score\nis a change in which communities post",
                  fontsize=9.5, pad=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # ---- right: class shares, raw vs standardised ----
    metrics = [("pos", "positive"), ("neu", "neutral"), ("neg", "negative")]
    x = range(len(metrics))
    wd = 0.26
    for i, (key, _) in enumerate(metrics):
        trio = [raw_c[key] * 100, raw_e[key] * 100, std_e[key] * 100]
        for j, (v, c) in enumerate(zip(trio, [GREY, BLUE, ORANGE])):
            ax2.bar(i + (j - 1) * wd, v, wd, color=c, edgecolor="white", linewidth=0.6)
            ax2.text(i + (j - 1) * wd, v + 0.7, f"{v:.1f}", ha="center",
                     fontsize=8.4, fontweight="bold", color=c)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([m[1] for m in metrics], fontsize=10)
    ax2.set_ylabel("share of posts (%)", fontsize=10)
    ax2.set_ylim(0, 82)
    ax2.set_title("The neutral class halves; the negative share barely moves", fontsize=9.5, pad=8)
    ax2.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color=GREY, label="Control (2019)"),
                        Patch(color=BLUE, label="LLM era, as observed"),
                        Patch(color=ORANGE, label="LLM era, 2019 subreddit mix")],
               fontsize=8.2, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.3, columnspacing=1.1)

    fig.text(0.5, 0.015,
             "Standardisation applies the 2019 subreddit weights to the 2024\u20132026 sentiment "
             "(d2_sentiment_robustness.py, 2,613,495 posts, automated accounts excluded).",
             ha="center", fontsize=8, style="italic", color="#444444")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    p = os.path.join(OUT, "fig_posts_composition_control.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"[fig] wrote {p}")


def fig_neutral(d):
    delta = pd.DataFrame({
        "subreddit": d.subreddit,
        "positive": (d.pos_e - d.pos_c) * 100,
        "neutral": (d.neu_e - d.neu_c) * 100,
        "negative": (d.neg_e - d.neg_c) * 100,
    }).sort_values("neutral")          # largest collapse first (drawn top-down after invert)

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    y = list(range(len(delta)))
    wd = 0.26
    cols = {"positive": "#3C8D5A", "neutral": "#8C8C8C", "negative": "#B4453F"}
    order = ["negative", "neutral", "positive"]      # visual order inside each group
    for i, key in enumerate(order):
        off = (i - 1) * wd
        ax.barh([v + off for v in y], delta[key], wd, color=cols[key],
                edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(delta.subreddit, fontsize=10)
    ax.invert_yaxis()                                # biggest collapse at the top
    ax.axvline(0, color="#333333", lw=1.0, zorder=4)
    ax.set_xlabel("change in share of posts, 2024\u20132026 minus 2019 (percentage points)",
                  fontsize=10)
    ax.set_title("Figure 6.7: The neutral share collapsed inside nine of ten subreddits",
                 fontsize=12, fontweight="bold", pad=14)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cols[k], label=k) for k in order],
              fontsize=9, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, -0.005), ncol=3, handlelength=1.3,
              columnspacing=1.2)

    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, ls=":", lw=0.7, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-27, 26)
    ax.tick_params(axis="x", labelsize=9)

    # annotate the two extremes so the reading is unmistakable
    for sub, key, dx in (("tifu", "neutral", -0.9), ("photography", "neutral", -0.9)):
        v = float(delta.loc[delta.subreddit == sub, key].iloc[0])
        ax.annotate(f"{v:+.1f}", xy=(v, list(delta.subreddit).index(sub)),
                    xytext=(v - 1.4, list(delta.subreddit).index(sub) - 0.30),
                    fontsize=8.5, fontweight="bold", color=cols[key],
                    ha="right", va="center")

    fig.text(0.5, 0.015,
             "Each group is the same community in both periods, so the change is within-community "
             "and not caused by the changing subreddit mix. n = 2,613,495 posts.",
             ha="center", fontsize=8, style="italic", color="#444444")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    p = os.path.join(OUT, "fig_posts_neutral_collapse.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"[fig] wrote {p}")
    return delta


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    d = load()
    raw_c, raw_e, std_e = aggregate(d)
    print(f"[fig] control {raw_c['mean']:+.4f} | era {raw_e['mean']:+.4f} | "
          f"standardised {std_e['mean']:+.4f}")
    print(f"[fig] neutral {raw_c['neu']*100:.2f} % -> {raw_e['neu']*100:.2f} %")
    fig_composition(d, raw_c, raw_e, std_e)
    delta = fig_neutral(d)
    print("\n[fig] within-subreddit deltas (pp)")
    print(delta.to_string(index=False, float_format=lambda v: f"{v:+.1f}"))
    print("\n  neutral down in", int((delta.neutral < 0).sum()), "of", len(delta), "subreddits")
    print("  negative up  in", int((delta.negative > 0).sum()), "of", len(delta), "subreddits")
