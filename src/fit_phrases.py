import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent.parent
conn = sqlite3.connect(ROOT / "data" / "reviews.db")
df = pd.read_sql("SELECT review_id, review_text, recommended FROM reviews", conn)
conn.close()

text = df["review_text"].str.lower()
overall = df["recommended"].mean()

NEG = r"(?<!n't )(?<!not )"
ADV = r"(?:very |really |quite |a bit |a little |somewhat )?"
PHRASES = {
    "too small": r"too small",
    "too tight": r"too tight",
    "too short": r"too short",
    "too big": r"too big",
    "too large": r"too large",
    "too long": r"too long",
    "runs small": NEG + r"runs? " + ADV + r"small",
    "runs large": NEG + r"runs? " + ADV + r"(?:large|big)",
    "size up": r"size[d]? up",
    "size down": r"size[d]? down",
    "true to size": r"true to size",
}


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


rows = []
for name, pattern in PHRASES.items():
    mask = text.str.contains(pattern, regex=True)
    n = int(mask.sum())
    if n < 20:
        continue
    k = int(df.loc[mask, "recommended"].sum())
    n_o = len(df) - n
    k_o = int(df["recommended"].sum()) - k
    low, high = wilson(k, n)
    p = chi2_contingency([[k, n - k], [k_o, n_o - k_o]])[1]
    rows.append({
        "phrase": name,
        "reviews": n,
        "recommend_pct": round(100 * k / n, 1),
        "ci_low": round(100 * low, 1),
        "ci_high": round(100 * high, 1),
        "others_pct": round(100 * k_o / n_o, 1),
        "p_value": float(f"{p:.3g}"),
    })

table = pd.DataFrame(rows).sort_values("recommend_pct")
alpha = 0.05 / len(PHRASES)
table.to_csv(ROOT / "results" / "fit_phrases.csv", index=False)

print(f"Overall recommend rate: {100 * overall:.1f}%")
print(f"Phrases tested: {len(PHRASES)}. Strict cut-off for 'significant' (Bonferroni): p < {alpha:.4f}\n")
print(table.to_string(index=False))

for name in table[table["reviews"] >= 50].head(2)["phrase"]:
    print(f"\nContext check for '{name}' (5 random matches)")
    pat = re.compile(PHRASES[name])
    hits = df[text.str.contains(PHRASES[name], regex=True)].sample(5, random_state=1)
    for _, r in hits.iterrows():
        t = r["review_text"].lower()
        m = pat.search(t)
        start = max(0, m.start() - 70)
        print(f"- review {r['review_id']} (customer recommended={r['recommended']}) ...{t[start:m.end() + 70]}...")