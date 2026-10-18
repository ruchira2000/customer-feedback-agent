import sys
from pathlib import Path

import pandas as pd

if len(sys.argv) < 2:
    sys.exit("Usage: python tally_verdicts.py <model-name>")

MODEL = sys.argv[1]
ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "results" / f"disagreements_{MODEL}.csv"

try:
    df = pd.read_csv(path)
except UnicodeDecodeError:
    df = pd.read_csv(path, encoding="cp1252")

allowed = ["ai error", "mixed review", "tick looks wrong"]
df["verdict"] = df["my_verdict"].fillna("").astype(str).str.strip().str.lower()

print(f"File: {path.name}  |  rows: {len(df)}")

blank = df[df["verdict"] == ""]
if len(blank):
    print(f"Still blank: {len(blank)} rows (review ids: {blank['review_id'].tolist()})")

unknown = df[(df["verdict"] != "") & (~df["verdict"].isin(allowed))]
for _, r in unknown.iterrows():
    print(f"Not one of the three allowed phrases: review {r['review_id']} -> '{r['my_verdict']}'")

print("\nTally:")
for label in allowed:
    print(f"  {label}: {int((df['verdict'] == label).sum())}")