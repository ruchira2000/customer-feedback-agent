import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
MODEL = "gemini-3.5-flash-lite"


def read_json(name):
    path = RES / name
    if not path.exists():
        print("  missing:", name)
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(name):
    path = RES / name
    if not path.exists():
        print("  missing:", name)
        return []
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")
    df = df.astype(object).where(df.notna(), None)
    return df.to_dict(orient="records")


data = {
    "model": MODEL,
    "replay_before": read_json("replay_v1_before_fix.json"),
    "replay_after": read_json("replay.json"),
    "retest_before": read_json("retest_before.json"),
    "retest_after": read_json("retest_after.json"),
    "retest_fresh": read_json("retest_fresh.json"),
    "scores": read_csv("scores_" + MODEL + ".csv"),
    "disagreements": read_csv("disagreements_" + MODEL + ".csv"),
    "phrases": read_csv("fit_phrases.csv"),
}

if data["replay_before"] == data["replay_after"]:
    print("\nNote: replay.json is identical to the pre-fix recording. Re-record it with the fixed agent first.")

(DOCS / "site_data.json").write_text(
    json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
)
shutil.copy(ROOT / "data" / "reviews.db", DOCS / "reviews.db")

print("\nWrote docs/site_data.json")
for key, value in data.items():
    if isinstance(value, list):
        print("  " + key + ":", len(value), "items")
print("Copied docs/reviews.db:", round((DOCS / "reviews.db").stat().st_size / 1e6, 1), "MB")