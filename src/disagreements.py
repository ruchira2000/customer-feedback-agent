import sqlite3
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

if len(sys.argv) < 2:
    sys.exit("Usage: python disagreements.py <model-name>")

MODEL = sys.argv[1]
ROOT = Path(__file__).resolve().parent.parent

conn = sqlite3.connect(ROOT / "data" / "reviews.db")
df = pd.read_sql("SELECT review_id, review_text, recommended FROM reviews", conn)
conn.close()

llm = pd.read_csv(ROOT / "results" / f"llm_labels_{MODEL}.csv")
train, _ = train_test_split(df, test_size=0.2, random_state=42, stratify=df["recommended"])

sample = df.merge(llm[["review_id", "llm_answer"]], on="review_id")
sample = sample[sample["llm_answer"] != "UNPARSED"].reset_index(drop=True)
sample["llm_pred"] = sample["llm_answer"].map({"YES": 1, "NO": 0})

vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
baseline.fit(vec.fit_transform(train["review_text"]), train["recommended"])
sample["base_pred"] = baseline.predict(vec.transform(sample["review_text"]))

llm_right = sample["llm_pred"] == sample["recommended"]
base_right = sample["base_pred"] == sample["recommended"]

only_llm = int((llm_right & ~base_right).sum())
only_base = int((~llm_right & base_right).sum())
both_wrong = int((~llm_right & ~base_right).sum())
both_right = int((llm_right & base_right).sum())

print(f"Both right: {both_right} | only {MODEL} right: {only_llm} | only baseline right: {only_base} | both wrong: {both_wrong}")
if only_llm + only_base > 0:
    p = binomtest(only_llm, only_llm + only_base, 0.5).pvalue
    print(f"Paired test (exact McNemar) p-value: {p:.3f}  (below 0.05 would suggest a real difference)")

wrong = sample[~llm_right]
out = pd.DataFrame({
    "review_id": wrong["review_id"],
    "customer_said": wrong["recommended"].map({1: "YES", 0: "NO"}),
    "model_said": wrong["llm_answer"],
    "review_text": wrong["review_text"],
    "my_verdict": "",
})
out_path = ROOT / "results" / f"disagreements_{MODEL}.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved {len(out)} disagreements to results/{out_path.name}")