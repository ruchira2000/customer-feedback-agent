import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from sklearn.model_selection import train_test_split

load_dotenv()

if len(sys.argv) < 3:
    sys.exit("Usage: python evaluate_llm.py <model-name> <sample-size> [pause-seconds]")

MODEL = sys.argv[1]
SAMPLE_SIZE = int(sys.argv[2])
PAUSE_SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 13

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "reviews.db"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / f"llm_labels_{MODEL}.csv"

PROMPT = """You are reading a customer review of a clothing item.
Decide whether this customer would recommend the item to others.
Answer with exactly one word: YES or NO.

Review:
{text}"""

gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


class DailyLimit(Exception):
    pass


conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT review_id, review_text, recommended FROM reviews", conn)
conn.close()

# Same split the baseline model uses, so the AI only sees reviews the baseline never trained on.
# Shuffled once with a fixed seed, so every model gets the same reviews in the same order.
_, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["recommended"])
sample = test.sample(frac=1, random_state=42).head(SAMPLE_SIZE)

done = set()
if OUT_PATH.exists():
    done = set(pd.read_csv(OUT_PATH)["review_id"])


def ask_llm(text):
    waits = [20, 40, 80, 120]
    for attempt in range(len(waits) + 1):
        try:
            response = gemini.models.generate_content(
                model=MODEL,
                contents=PROMPT.format(text=text),
                config=types.GenerateContentConfig(temperature=0),
            )
            return (response.text or "").strip()
        except errors.APIError as e:
            if e.code == 429 and "PerDay" in str(e):
                raise DailyLimit()
            retryable = e.code in (429, 500, 503, 504)
            if retryable and attempt < len(waits):
                print(f"  temporary error {e.code}, waiting {waits[attempt]} seconds...")
                time.sleep(waits[attempt])
            else:
                raise
    return ""


print(f"Model: {MODEL} | target: {SAMPLE_SIZE} reviews | already saved: {len(done)}")
count = 0
for row in sample.itertuples():
    if row.review_id in done:
        continue
    try:
        answer = ask_llm(row.review_text)
    except DailyLimit:
        print("\nDaily free limit reached for this model. Run the same command later, or pick another model.")
        break

    clean = answer.upper().strip().strip("*").strip()
    if clean.startswith("YES"):
        label = "YES"
    elif clean.startswith("NO"):
        label = "NO"
    else:
        label = "UNPARSED"

    pd.DataFrame([{
        "review_id": row.review_id,
        "true_recommended": row.recommended,
        "llm_answer": label,
        "raw_answer": answer,
        "model": MODEL,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }]).to_csv(OUT_PATH, mode="a", header=not OUT_PATH.exists(), index=False)

    count += 1
    print(f"{count} done (review {row.review_id}: {label})")
    time.sleep(PAUSE_SECONDS)

print(f"Finished this run. Results are in results/{OUT_PATH.name}")