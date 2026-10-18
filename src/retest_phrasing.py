import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from google.genai import errors

import agent

if len(sys.argv) < 2:
    sys.exit("Usage: python retest_phrasing.py <label>   (for example: before or after)")

LABEL = sys.argv[1]
ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / f"retest_{LABEL}.json"

# (question, exact phrase the user asked about, recommend % known to be correct from our own analysis)
TESTS = [
    ("Are reviews that say 'too short' less likely to be recommended?", "too short", "75.8"),
    ("What is the recommend rate among customers saying an item is too big?", "too big", "73.3"),
    ("Do reviews mentioning 'too long' have a different recommend rate from the rest?", "too long", "84.3"),
]


async def main():
    print("Model:", agent.MODEL, "| label:", LABEL)
    results = []
    for i, (question, phrase, expected) in enumerate(TESTS, start=1):
        print(f"\n[{i}/{len(TESTS)}] {question}")
        answer, trace = None, []
        for attempt in range(3):
            try:
                answer, trace = await agent.ask(question)
                break
            except errors.APIError as e:
                if e.code == 429 and "PerDay" in str(e):
                    print("  Daily limit reached. Stopping.")
                    OUT_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
                    return
                if e.code in (429, 500, 503, 504) and attempt < 2:
                    print(f"  temporary error {e.code}, waiting 60 seconds...")
                    await asyncio.sleep(60)
                else:
                    print("  failed:", e.code)
                    break

        searched = [c["args"].get("keyword", "").strip().lower() for c in trace if c["tool"] == "keyword_impact"]
        used_phrase = phrase in searched
        right_number = bool(answer) and expected in answer
        passed = used_phrase and right_number

        print("  keywords searched:", searched)
        print(f"  searched the exact phrase: {used_phrase} | reported {expected}%: {right_number} -> {'PASS' if passed else 'FAIL'}")

        results.append({
            "question": question,
            "target_phrase": phrase,
            "expected_pct": expected,
            "keywords_searched": searched,
            "answer": answer,
            "passed": passed,
            "model": agent.MODEL,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        OUT_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        await asyncio.sleep(12)

    print(f"\nPassed {sum(r['passed'] for r in results)} of {len(results)}")


asyncio.run(main())