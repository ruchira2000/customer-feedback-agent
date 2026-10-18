import re
import sqlite3
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent.parent
conn = sqlite3.connect(ROOT / "data" / "reviews.db")
df = pd.read_sql(
    "SELECT review_id, department, class_name, review_text, recommended FROM reviews", conn
)
conn.close()

# Crude detector for reviews that mention a sizing problem or sizing adjustment.
# It skips phrases right after "not" or "n't", but it cannot read every kind of negation.
FIT_PATTERN = re.compile(
    r"too (small|large|big|tight|short|long|loose|snug)"
    r"|(?<!n't )(?<!not )runs? (very |really |quite |a bit |a little |somewhat )?(small|large|big|tight|short|long)"
    r"|size (up|down)|sized (up|down)"
)
df["fit_mention"] = df["review_text"].str.lower().apply(lambda t: bool(FIT_PATTERN.search(t)))

fit = df[df["fit_mention"]]
other = df[~df["fit_mention"]]
fit_rate = fit["recommended"].mean()
other_rate = other["recommended"].mean()
gap = other_rate - fit_rate

table = [
    [int((fit["recommended"] == 1).sum()), int((fit["recommended"] == 0).sum())],
    [int((other["recommended"] == 1).sum()), int((other["recommended"] == 0).sum())],
]
p_value = chi2_contingency(table)[1]

print(f"Reviews analysed: {len(df)}")
print(f"Reviews mentioning fit: {len(fit)} ({100 * len(fit) / len(df):.1f}%)")
print(f"Recommend rate, fit reviews: {100 * fit_rate:.1f}%")
print(f"Recommend rate, other reviews: {100 * other_rate:.1f}%")
print(f"Gap: {100 * gap:.1f} percentage points (chi-square p-value: {p_value:.2g})")
print(f"Illustration: if fit reviews recommended at the same rate as others, "
      f"there would be about {round(len(fit) * gap)} fewer non-recommenders.")

rows = []
for name, g in df.dropna(subset=["class_name"]).groupby("class_name"):
    f = g[g["fit_mention"]]
    o = g[~g["fit_mention"]]
    if len(f) < 20 or len(o) < 20:
        continue
    class_gap = o["recommended"].mean() - f["recommended"].mean()
    rows.append({
        "class_name": name,
        "reviews": len(g),
        "fit_mentions": len(f),
        "fit_share_pct": round(100 * len(f) / len(g), 1),
        "recommend_pct_fit": round(100 * f["recommended"].mean(), 1),
        "recommend_pct_other": round(100 * o["recommended"].mean(), 1),
        "gap_points": round(100 * class_gap, 1),
        "excess_nonrecommenders": round(len(f) * class_gap),
    })

by_class = pd.DataFrame(rows).sort_values("excess_nonrecommenders", ascending=False)
out_dir = ROOT / "results"
out_dir.mkdir(exist_ok=True)
by_class.to_csv(out_dir / "fit_by_class.csv", index=False)

print("\nClasses ranked by excess non-recommenders (classes with at least 20 fit mentions):")
print(by_class.head(8).to_string(index=False))

print("\nSanity check: 10 random reviews the detector flagged as fit mentions")
for _, r in fit.sample(10, random_state=1).iterrows():
    print(f"- review {r['review_id']} | {r['review_text'][:150]}")