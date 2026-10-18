import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
records = json.loads((ROOT / "results" / "replay.json").read_text(encoding="utf-8"))

NUMBER = re.compile(r"\d[\d,]*\.?\d*")
ALLOWED = {150.0, 30.0}  # thresholds written into the system prompt and the tool


def numbers_in(text):
    found = set()
    for token in NUMBER.findall(text):
        cleaned = token.replace(",", "").rstrip(".")
        try:
            found.add(float(cleaned))
        except ValueError:
            pass
    return found


total_flagged = 0
for i, rec in enumerate(records, start=1):
    answer = re.sub(r"(?m)^\s*\d+[.)]\s+", "", rec["answer"])  # ignore list numbering
    tool_numbers = numbers_in(json.dumps([c["result"] for c in rec["tool_calls"]]))
    missing = sorted(n for n in numbers_in(answer) if n not in tool_numbers and n not in ALLOWED)
    total_flagged += len(missing)

    print(f"[{i}] {rec['question'][:70]}")
    print(f"    tool calls: {len(rec['tool_calls'])} | model: {rec['model']}")
    print(f"    numbers not found in tool results: {missing if missing else 'none'}")

print(f"\nTotal flagged numbers: {total_flagged}")
print("A flagged number is not automatically wrong (it may be a calculation), but check it by hand.")