# Does the AI Analyst Actually Get It Right?

**[Explore the recorded runs and results](https://ruchira2000.github.io/customer-feedback-agent/)**

I built an AI agent that answers questions about 22,641 clothing reviews, then spent most of the project trying to catch it being wrong. This repo has the agent, the tests I ran on it, what I found, and a small website for exploring all of it without installing anything.

## Try it in 60 seconds

| Way in | What you need | What you get |
|---|---|---|
| **Watch the agent** | A browser | Six recorded conversations, before and after I fixed a bug, with every tool call visible |
| **Read the evidence** | A browser | The scoreboard, my audit of 13 disagreements, the before-and-after test, the sizing analysis |
| **Use the toolbox yourself** | A browser | The agent's five tools running the same SQL on the real reviews, inside your page. No server, no key, no AI |
| **Run the real agent** | Python and a free Google AI Studio key | Ask your own questions (steps below) |

The website only shows recorded output and a database that runs in your browser. That is deliberate. Free AI services change often, and I wanted the data, the tools and the results to keep working without them. Only the "thinking" needs an AI.

## Five things I found

1. **The AI beat a classic ML model, but not convincingly.** On 200 held-out reviews it scored 93.5% accuracy against 89.0%, but a paired test gives p = 0.064. Suggestive, not proven.
2. **Most disagreements were not AI mistakes.** The AI disagreed with the customer on 13 reviews. I read all 13. One was a clear AI error, 8 were mixed reviews where either answer is defensible, and 4 looked like the customer's own tick contradicted what they wrote.
3. **My agent answered the wrong question, with real numbers.** Asked about reviews saying "too small", it searched the broader word "small" and reported 80.1% recommended. The true figure for "too small" is 65.2%. A check that every number appears in the tool results passed this answer, so I wrote a test for the keyword searched as well. One tool-description change and one added rule moved it from 2 of 5 to 5 of 5, and 3 of 3 on questions I had not tuned on.
4. **Fixing one thing changed another.** In the re-recorded runs, asked for the lowest classes "excluding tiny groups", the fixed agent listed Trend (118 reviews) first, with a small-sample warning, where the original left it out. One answer also leaked its own reasoning ("Wait, 224 is greater than 150..."). I stopped tuning there, because every tweak needs both test sets re-run and I would be tuning to my own test.
5. **A null result hid a real one.** Reviews that mention fit in any way are recommended 82.3% of the time, against 81.8% for the rest (p = 0.41). Split by exact phrase, two opposite effects cancel out. Complaints like "too small" go with lower recommendation, and size advice like "size down" goes with higher.

## How the agent works

```
22,641 reviews (SQLite, opened read-only)
        |
   MCP server: 5 tools (overview, department stats, class stats, keyword impact, review search)
        |
   Gemini agent loop (written by hand, capped at 8 steps)
        |
   Answer, plus a recorded trace of every tool call
```

The agent can only answer through its tools. It cannot write to the database. Its instructions tell it to quote exact figures, warn about samples under 150 reviews, never guess, and say so when the data cannot answer.

## Test 1: does the AI judge reviews like customers do?

Each review comes with the customer's own yes or no to "would you recommend this?". I hid that answer, showed the AI only the review text, and compared its guess with the real answer. I also ran a classic TF-IDF plus logistic regression model and a lazy baseline that always says "recommended".

200 reviews from a held-out 20% test split, the same reviews for every method, temperature 0, model gemini-3.5-flash-lite. 34 of the 200 customers did not recommend.

| Method | Accuracy | Unhappy customers caught | When it flags "not recommended", how often is it right |
|---|---|---|---|
| Always say "recommended" | 83.0% | 0% | never flags anyone |
| TF-IDF + logistic regression | 89.0% | 97.1% | 61.1% |
| gemini-3.5-flash-lite | 93.5% | 97.1% | 73.3% |

Both real methods caught 33 of the 34 unhappy customers. Gemini made 13 errors and the baseline made 22. On the 19 reviews where they differed, Gemini was right on 14 and the baseline on 5 (exact paired test, p = 0.064). The baseline is deliberately tuned to catch unhappy customers, which lowers its precision, so the comparison depends on that choice.

## Test 2: who is wrong when they disagree?

I read all 13 disagreements and sorted them with one written rule.

| Verdict | Count |
|---|---|
| Mixed review (real pros and cons, either answer is defensible) | 8 |
| The customer's tick looks wrong given what they wrote | 4 |
| Clear AI error | 1 |

In 12 of the 13, the customer ticked "recommended" but wrote complaints, mostly about fit or quality. The AI holds reviews to a stricter standard than customers do. That makes it useful for flagging reviews for a human to read, and unreliable as the final judge. I only audited the disagreements, not the 187 reviews where the AI and customer agreed, so I do not report a "corrected accuracy". That would favour the AI.

## Test 3: catching the agent answering the wrong question

I recorded six conversations, including deliberate traps. The agent refused to delete data, refused to invent sales figures when told to ignore its instructions, and said plainly that the data has no order values. A script confirmed every number in every answer appears in the tool results.

One failure slipped past that check: the "too small" answer above. So I tested five differently worded questions before and after the fix.

| | Searched the exact phrase and quoted the right figure |
|---|---|
| Before the fix | 2 of 5 |
| After the fix | 5 of 5 |
| 3 new questions the fix was not tuned on | 3 of 3 |

I wrote the fix after seeing which questions failed, so this shows the change helped, not that the problem is solved. The test checks the keyword searched and the figure quoted, not whether the agent interprets a gap sensibly.

## What the reviews say about sizing

| Phrase | Reviews | Recommended |
|---|---|---|
| "too small" | 224 | 65.2% |
| "too large" | 186 | 68.8% |
| "too big" | 558 | 73.3% |
| "too short" | 525 | 75.8% |
| all reviews | 22,641 | 81.9% |
| "runs large" | 1,007 | 88.3% |
| "size down" | 897 | 88.7% |
| "size up" | 674 | 89.2% |
| "true to size" | 1,287 | 92.6% |

Eight of the 11 phrases I tested pass a strict cut-off that allows for testing 11 at once. These are associations, not proof that fit causes lost recommendations. There is no revenue in this dataset, so I measure impact in reviews, not dollars.

**What I would recommend:** look first at products with the most "too small", "too large" and "too big" reviews, and test clearer size guidance on them as an experiment. Use the AI to triage reviews for a human to read, not to decide.

## Governance: what keeps the agent in check

- **Read-only access.** The database opens in read-only mode, so the agent cannot change or delete data.
- **No SQL injection.** Every query passes values separately from the SQL text.
- **Capped output.** Searches return at most 10 reviews, and the agent loop stops after 8 steps.
- **Rules in the instructions.** Quote exact figures, warn about samples under 150 reviews, never guess, and say so when the data cannot answer.
- **An audit trail.** Every tool call and result is recorded, and a script checks every number in an answer against the tool results.
- **Public data only.** The reviews are public domain and anonymised. Google's free tier may use submitted data to train models, so never send private customer data this way.

Known gaps: the agent warns about small samples but not about small differences between groups, so a class ranked "lowest" may only be 2 points below average. I tried only one prompt-injection attempt.

## Honest limits

- Only 34 of the 200 reviews came from unhappy customers, so the intervals are wide.
- One model, one prompt, one run each. Temperature was 0, but identical reruns are not guaranteed.
- The customer's tick is the answer key and it is noisy, as the audit showed.
- The audit was by one person, who started from an AI-suggested draft and changed one label. Some reviews in the dataset are cut off mid-sentence.
- The sizing phrase detector cannot read negation ("not too large" still matches). I chose "too small" after seeing earlier output, so treat the phrase analysis as exploratory.
- One retailer, women's clothing, older data.
- Not built: a fully offline mode with a local model. Small local models call tools less reliably than large ones, so I would test it with the same before-and-after questions.

## Notes on running this for free

In October 2026 the free Gemini tiers differed a lot by model. One Flash model allowed 20 requests a day, while the Flash-Lite models allowed 500 a day. A model I first picked had closed to new users, and a key from AI Studio was rejected until I created a new one in a new project. So every result records its exact model name, and the code takes the model as a setting.

## Run it yourself

```
git clone https://github.com/ruchira2000/customer-feedback-agent.git
cd customer-feedback-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Mac or Linux, activate with `source venv/bin/activate`. Create a free key in Google AI Studio and put it in a file called `.env`:

```
GEMINI_API_KEY=your_key_here
```

Then, from the `src` folder:

```
python load_data.py
python agent.py
```

To reproduce the evaluation:

```
python evaluate_llm.py gemini-3.5-flash-lite 200 5
python score.py gemini-3.5-flash-lite
python disagreements.py gemini-3.5-flash-lite
```

To rebuild and view the website locally:

```
python export_site_data.py
cd ..\docs
python -m http.server 8000
```

Then open http://localhost:8000. If a model name has been retired by the time you read this, change `MODEL` in `agent.py` and pass a current name to the evaluation scripts.

## Project layout

```
customer-feedback-agent/
├── data/reviews.csv        the public domain reviews
├── docs/                   the website (served free by GitHub Pages)
├── results/                labels, scores, audit, recorded conversations, tests
└── src/
    ├── load_data.py        reviews into SQLite
    ├── server.py           the MCP toolbox
    ├── agent.py            the agent loop
    ├── evaluate_llm.py     the AI labels 200 reviews
    ├── score.py            AI vs baseline vs lazy guess
    ├── disagreements.py    reviews where AI and customer differ
    ├── tally_verdicts.py   counts my hand verdicts
    ├── business_analysis.py, fit_phrases.py       the sizing analysis
    ├── record_replay.py, check_grounding.py, retest_phrasing.py       agent testing
    └── export_site_data.py packs results for the website
```

Reviews: Women's E-Commerce Clothing Reviews, public domain (CC0).

## Author

**Ruchira Surendra**, Data Analyst, Melbourne · [LinkedIn](https://www.linkedin.com/in/ruchira-surendra)