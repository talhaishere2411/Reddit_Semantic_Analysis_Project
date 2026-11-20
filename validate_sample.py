"""Check the AI-likeness heuristic against human judgement.
Step 1 (export):  python validate_sample.py export "data/ai/comments/*.jsonl" 200
Step 2: open output_ai/validation_sample.csv, fill the human_label column with AI or HUMAN for every row.
Step 3 (score):  python validate_sample.py score
Raw text is exported only for manual labelling; keep that file private and do not publish it."""
import sys, glob, json, csv, random
import pandas as pd

def load(path):
    rows = []
    for f in glob.glob(path):
        with open(f, encoding="utf8") as fh:
            for line in fh:
                try: rows.append(json.loads(line))
                except Exception: pass
    return pd.DataFrame(rows)

if sys.argv[1] == "export":
    import pyarrow  # noqa: F401  (needed to read the scored parquet)
    n = int(sys.argv[3])
    scored = pd.read_parquet("output_ai/comments_scored.parquet")
    scored = scored[scored.period == "llm_era"][["id", "ai_like", "ai_score"]]
    raw = load(sys.argv[2])[["id", "body"]]
    m = scored.merge(raw, on="id")
    pos = m[m.ai_like == 1].sample(min(n, (m.ai_like == 1).sum()), random_state=42)
    neg = m[m.ai_like == 0].sample(min(n, (m.ai_like == 0).sum()), random_state=42)
    out = pd.concat([pos, neg]).sample(frac=1, random_state=1)
    out["human_label"] = ""
    out.rename(columns={"body": "text"})[["id", "text", "ai_like", "human_label"]].to_csv("output_ai/validation_sample.csv", index=False, quoting=csv.QUOTE_ALL)
    print("Exported", len(out), "rows. Fill human_label with AI or HUMAN.")
else:
    d = pd.read_csv("output_ai/validation_sample.csv")
    d = d[d.human_label.isin(["AI", "HUMAN"])]
    flagged = d[d.ai_like == 1]; unflagged = d[d.ai_like == 0]
    precision = (flagged.human_label == "AI").mean() if len(flagged) else float("nan")
    missed = (unflagged.human_label == "AI").mean() if len(unflagged) else float("nan")
    print({"labelled": len(d), "flagged_labelled": len(flagged), "unflagged_labelled": len(unflagged),
           "precision_of_flagged": round(float(precision), 3),
           "share_of_unflagged_that_is_AI": round(float(missed), 3)})
    print("The sample is balanced (equal numbers flagged and not flagged), so only these two rates are valid; do not report recall from it.")
