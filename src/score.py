import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

if len(sys.argv) < 2:
    sys.exit("Usage: python score.py <model-name>")

MODEL = sys.argv[1]
ROOT = Path(__file__).resolve().parent.parent

conn = sqlite3.connect(ROOT / "data" / "reviews.db")
df = pd.read_sql("SELECT review_id, review_text, recommended FROM reviews", conn)
conn.close()

llm = pd.read_csv(ROOT / "results" / f"llm_labels_{MODEL}.csv")
print(f"Model: {MODEL}")
print(f"Reviews labeled: {len(llm)}   Unparsed answers: {(llm['llm_answer'] == 'UNPARSED').sum()}")

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["recommended"])

sample = df.merge(llm[["review_id", "llm_answer"]], on="review_id")
assert sample["review_id"].isin(test["review_id"]).all(), "sample must come from the test split"
sample = sample[sample["llm_answer"] != "UNPARSED"].reset_index(drop=True)

y_true = sample["recommended"]
llm_pred = sample["llm_answer"].map({"YES": 1, "NO": 0})
lazy_pred = pd.Series(1, index=sample.index)

vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
baseline.fit(vec.fit_transform(train["review_text"]), train["recommended"])
baseline_pred = baseline.predict(vec.transform(sample["review_text"]))


def report(name, pred):
    return {
        "method": name,
        "accuracy": accuracy_score(y_true, pred),
        "precision_not_rec": precision_score(y_true, pred, pos_label=0, zero_division=0),
        "recall_not_rec": recall_score(y_true, pred, pos_label=0, zero_division=0),
        "f1_not_rec": f1_score(y_true, pred, pos_label=0, zero_division=0),
    }


results = pd.DataFrame([
    report("Always say recommended", lazy_pred),
    report("TF-IDF + Logistic Regression", baseline_pred),
    report(MODEL, llm_pred),
]).round(3)

n = len(sample)
acc = results.iloc[2]["accuracy"]
margin = 1.96 * np.sqrt(acc * (1 - acc) / n)
print(f"\nScored on {n} reviews ({(y_true == 0).sum()} not recommended)")
print(results.to_string(index=False))
print(f"\nRough 95% margin of error on accuracy: +/- {margin * 100:.1f} points")

results.to_csv(ROOT / "results" / f"scores_{MODEL}.csv", index=False)

wrong = sample[llm_pred != y_true]
print(f"\n{MODEL} disagreed with the customer on {len(wrong)} reviews. First 5:")
for _, r in wrong.head(5).iterrows():
    print(f"- review {r['review_id']} | customer said {'YES' if r['recommended'] == 1 else 'NO'}, "
          f"model said {r['llm_answer']} | {r['review_text'][:160]}")