<!-- PROJECT LOGO -->
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Logo" width="120" height="120">
</p>

<h1 align="center">SMS Recruiting Assistant</h1>

<p align="center">
  A multi-agent SMS screening bot for a Python Developer role<br>
  Technion · Generative AI &amp; LLM Hands-On (HTCGAI4) · Final Project
</p>

---
<br></br>

## Table of Contents

- [About The Project](#about-the-project)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Code Examples](#code-examples)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [To-Do List](#to-do-list)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---
<br></br>

## About The Project

> An SMS assistant that screens candidates for a Python Developer opening: it
> answers their questions about the role, offers interview slots that are
> genuinely free, and closes the conversation — whether the candidate books an
> interview or withdraws.

A **Main Agent** reads the conversation so far and decides what happens next.
It routes each turn to one of three advisors:

```text
        candidate SMS
              |
          Main Agent
         /     |      \
 continue   schedule    end
     |         |         |
Info        Scheduling  Exit
Advisor     Advisor     Advisor
(Chroma +   (SQLite +   (fine-tuned
 job spec)   function    model)
             calling)
```

<div style="background: #272822; color: #f8f8f2; padding: 10px; border-radius: 8px;">
  <b>Technologies:</b> Python, LangChain, OpenAI API, Chroma, SQLite, Streamlit, scikit-learn
</div>

---
<br></br>

## Features

- [x] Multi-agent orchestration — one router, three specialised advisors
- [x] Four prompt versions, scored against each other in one eval run
- [x] Retrieval-augmented answers over the job description (Chroma)
- [x] Function calling against a real interview calendar (SQLite)
- [x] Relative dates (“next Friday”) resolved from the conversation's own clock
- [x] Closed days handled explicitly instead of returning nothing
- [x] Fine-tuned Exit Advisor for candidate withdrawals (LoRA, local), measured against the base model
- [x] Evals with a baseline, a ceiling, and a confusion matrix
- [x] Streamlit interface (deployment to Streamlit Community Cloud pending)

---
<br></br>

## Getting Started

### Prerequisites

- Python >= 3.10
- An OpenAI API key

### Installation

```bash
git clone https://github.com/OlegRogazkinMedicalDevice/genai-recruiter-bot.git
cd genai-recruiter-bot

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then put your key in .env

python scripts/check_setup.py      # says what is still missing
```

### One-time data build

```bash
python scripts/build_schedule_db.py data/schedule.db   # calendar  (no API key needed)
python scripts/build_chroma.py                         # embeddings (uses the API)
python scripts/build_finetune_data.py                  # training file for the Exit Advisor
```

Fine-tuning (after the data build, needs `requirements-train.txt`):

```bash
pip install -r requirements-train.txt
python scripts/finetune_exit_advisor.py
# then put LOCAL_EXIT_MODEL=models/exit-advisor-lora in .env
```

`data/schedule.db` and `chroma_store/` are committed to the repository, so the
deployed app never rebuilds them.

---
<br></br>

## Usage

In the terminal:

```bash
python -m app.main
```

In the browser:

```bash
streamlit run streamlit_app/streamlit_main.py
```

Deployed: **&lt;paste the Streamlit Community Cloud URL here&gt;**
(the key goes in *Settings → Secrets*, never in the repository).

---
<br></br>

## Evaluation

`tests/test_evals.ipynb` measures the Main Agent against the 59 human-labelled
recruiter turns in `sms_conversations.json`. The figures below are from the run
of 27 November 2026 that also exercised the fine-tuned Exit Advisor, so every
number in this README, in `tests/eval_summary.md` and in the committed notebook
comes from one execution on one machine. `gpt-4.1-mini` at temperature 0.

| Approach | Accuracy |
|---|---|
| Random — 1 of 3 | 33.33% |
| Baseline — always `continue` | 42.37% |
| Main Agent, prompt **v1** — the obvious reading of the labels | 49.15% |
| Best possible from the turn number alone | 71.19% |
| Main Agent, prompt **v3** — v2 plus paraphrased few-shot | 72.88% |
| Main Agent, prompt **v4** — v2 plus a hard output-format block | 77.97% |
| Main Agent, prompt **v2** — plus the decision rules in the data | **79.66%** |
| Rules, no model — `app/modules/rule_reference.py` | 88.14% |
| Ceiling — contradictory labels | 96.61% |

`tests/eval_summary.md` is written by the notebook itself, so these figures are
never transcribed by hand.

### What the measurement actually showed

**The prompt was worth 30 points.** v1 states the three labels the obvious way
and scores 49.15%. v2 adds seven decision rules derived from the labelled data —
*the candidate's first real answer about experience is already `schedule`*, *a
time the candidate accepts is `end`, not `schedule`*, *declining a slot is
`schedule`, not `end`* — and scores 79.66%. The rules came out of the errors
made by the regex reference, not out of intuition.

**Then two further elaborations of v2 both made it worse.** This is the part
worth keeping.

*v3 adds six hand-written few-shot examples* and scores 72.88% — 6.78% below
v2, four items out of 59, past this notebook's own two-item noise threshold.
Each example is a single candidate message, while the labels in this data depend
on the whole conversation, so isolated one-liners taught the model to decide
from the last message alone.

*v4 adds a hard output-format block* — "exactly one word, you are not the
recruiter, there is no fourth option" — aimed squarely at the three items where
the model answers in character instead of emitting a label. It scores 77.97%:
one item below v2, inside the noise band, so it bought nothing and it did not
remove the `invalid` answers either.

Two independent attempts to improve a working prompt both failed to beat it.
Elaboration is not improvement, and without the eval both v3 and v4 would have
looked like progress — they are longer, more explicit, and more professional to
read. **v2 is what the app runs**, and it is also the cheapest of the four.

**The measurement is less reproducible than `temperature=0` suggests.** The
same four prompts were run three times against the same 59 items with no code
change in between — twice on one machine, once on another. v2 came out at
83.05%, 81.36% and 79.66%; v1 at 52.54%, 49.15% and 49.15%. That is a spread of
two items on v1 and three on v2, and the ordering of the four prompts never
changed. `temperature=0` makes a single run repeatable within itself; it does
not make two environments agree, because the provider is free to serve a
different snapshot behind the same model name. The practical consequence is
that the two-item noise rule above is a floor, not a ceiling: a difference
worth acting on has to survive a re-run, and a prompt change should be measured
in the same session as the baseline it is compared against.

**Seven regular expressions still beat the language model by eight points.** On
this task the model is not earning its cost in accuracy; what it buys is
handling phrasings nobody wrote a rule for. That is worth saying plainly rather
than hiding behind the headline number.

**Most of the signal is positional.** Knowing only how deep into the
conversation a turn is — and nothing of what was said — gets 71.19%. The model's
79.66% is eight points above that, and that gap, not the 37-point gap over
"always continue", is what the model actually earned.

### Where the 12 remaining failures come from

| Count | Cause |
|---|---|
| 5 | The labels contradict themselves: the first experience answer is `schedule` in ten conversations and `continue` in five (2, 3, 4, 10, 11). A rule can only take the majority, so these are lost either way. |
| 3 | The model answered as the recruiter instead of emitting a label (conversations 10 and 11), and `normalize()` scored it `invalid`. Earlier it returned the raw text, which entered the confusion matrix as a phantom label — an eval bug that flattered the model. Counting these as mistakes is the honest reading, so the 79.66% already carries them. Prompt v4 was written to remove them and did not. |
| 3 | The candidate names a time and the data calls it `end`, but the model keeps the conversation open: *"Monday at 3 PM is good"* (conversations 3 and 10) and *"Tuesday at 10 AM works"* (conversation 14). Prompt rule 2 covers exactly this and the model still misses it — these three are the whole of the drop in `end` recall, from 0.933 in the earlier run to 0.800 here. |
| 1 | Genuinely ambiguous: conversation 8, turn 7 reads as a soft refusal and is labelled `schedule`. |

Per-label recall on the winning prompt: `schedule` 0.842, `end` 0.800,
`continue` 0.760. Precision runs the other way — `end` 0.923, `continue` 0.905,
`schedule` 0.727 — which says the model reaches for `schedule` when unsure.

---
<br></br>

## Code Examples

Ask the calendar directly:

```python
from app.modules import db

db.find_available_slots("2024-04-03T15:12:00Z", limit=3)
# [{'date': '2024-04-04', 'time': '09:00', 'weekday': 'Thursday'}, ...]
```

Run one conversation:

```python
from app.main import RecruiterBot

bot = RecruiterBot(conversation_start="2024-04-03T15:12:00Z")
print(bot.opening()["text"])
print(bot.respond("I've used Python for five years, mostly data work.")["text"])
print(bot.respond("Can we do next Friday?")["text"])
```

---
<br></br>

## Project Structure

```text
genai-recruiter-bot/
├── app/
│   ├── __init__.py
│   ├── main.py                     # orchestration: RecruiterBot
│   └── modules/
│       ├── __init__.py
│       ├── config.py               # paths, models, prompts — single source of truth
│       ├── db.py                   # interview calendar (SQLite)
│       ├── main_agent.py           # continue / schedule / end
│       ├── info_advisor.py         # Chroma retrieval over the job description
│       ├── scheduling_advisor.py   # function calling against the calendar
│       └── exit_advisor.py         # closing message, fine-tuned
├── streamlit_app/
│   ├── __init__.py
│   └── streamlit_main.py
├── scripts/
│   ├── check_setup.py              # is everything in place? run this first
│   ├── build_schedule_db.py        # db_Tech.sql -> SQLite
│   ├── build_chroma.py             # job description -> embeddings
│   ├── build_finetune_data.py      # training + held-out validation JSONL
│   └── finetune_exit_advisor.py    # LoRA on Qwen2.5-0.5B-Instruct
├── models/
│   └── exit-advisor-lora/          # the fine-tuned adapter, a few MB
├── tests/
│   └── test_evals.ipynb
├── data/
│   ├── schedule.db                 # 9,396 slots, 2024
│   ├── sms_conversations.json      # 15 conversations, 59 labelled turns
│   ├── job_description.pdf
│   └── db_Tech.sql                 # the original script, for reference
├── chroma_store/                   # built by scripts/build_chroma.py
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example        # the shape of the Streamlit Cloud secrets
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---
<br></br>

## Design Decisions

**The calendar was converted from MS SQL Server to SQLite.** `db_Tech.sql`
builds a database on the developer's own machine. Streamlit Community Cloud
cannot reach it, so the deployed app would fail on the first scheduling turn.
`scripts/build_schedule_db.py` reproduces the same table — 2024, no Saturdays
and no Mondays, 09:00–17:00, four positions — as a file in the repository.

**Availability is generated with a fixed seed.** The original uses `NEWID()`,
so every run produces a different calendar and the evals stop being comparable.

**Slots are compared as timestamps, not dates.** Conversation 1 starts at
15:12; a date-only filter offers that candidate a 09:00 slot on the same day,
in the past.

**Relative dates resolve against `start_time_utc`.** The conversations are from
April 2024 and the calendar only covers 2024. Resolving “next Friday” against
the real today lands two years outside the data and returns nothing.

**Mondays are answered, not dropped.** Candidates ask for Mondays in
conversations 1, 3 and 10, and the calendar has none. The bot says so and
offers the nearest alternatives.

**Fine-tuning was applied to the Exit Advisor, not the Main Agent.** The label
`end` covers two opposite endings: 11 confirmed interviews and 4 withdrawals
(conversations 5, 8, 13, 15). A prompted model congratulates both. With only 4
refusals there is nothing to hold out, so the validation set in
`data/exit_advisor_valid.jsonl` is hand-written for this project — scoring on
the training refusals would measure memorisation.

**Fine-tuning runs locally on Hugging Face, not on OpenAI.** OpenAI's
fine-tuning platform is closed to new accounts — organisations that had not
run a job before 7 May 2026 can no longer create one. So
`scripts/finetune_exit_advisor.py` trains a LoRA adapter on
Qwen2.5-0.5B-Instruct instead. A full fine-tune of a 0.5B model needs around
8 GB of RAM for the optimiser states alone; LoRA trains 8.8M parameters — rank
16 across all seven projections, about 1.8% of the 494M in the base model —
runs on a laptop CPU, and produces an adapter small enough to commit. The
35,237,104-byte `adapter_model.safetensors` in `models/exit-advisor-lora/` is
exactly those 8.8M parameters in fp32 plus the safetensors header. The model
ends up as a folder in this repository rather than an id on someone else's
server.

**The fine-tuning worked, and it cost something.** Section 9 of the notebook
runs the base model and the adapter over the same four held-out cases. On the
two withdrawals the adapter does the thing it was trained for: the base model
answers a candidate who has taken another job with *"We wish you all the best
in your new role!"*, the tuned one does not congratulate anybody. But it
answers **both** withdrawals with the identical sentence — *"No worries—I
appreciate the update. Take care!"* — which is mode collapse, and with 4
refusals in 15 training examples it is the expected failure. On the two
confirmations it got worse rather than better: the base model keeps the agreed
time (*"confirmed for Thursday at 2 PM"*), the tuned one drops it, and on the
fourth case produces the broken *"Excellent, we'll be in. See you then!"*.
Fixing one behaviour and breaking the other is not an improvement, which is why
this is written down rather than summarised as a success. The honest reading is
that 15 examples are enough to move a 0.5B model's register and not enough to
teach it a task.

**Loss is computed on the reply only.** The prompt tokens are masked with
`-100`, so the model is scored on what the recruiter should write rather than
on reproducing the conversation it was handed. With 15 examples that is the
difference between learning the behaviour and memorising the inputs.

**torch and transformers are not in `requirements.txt`.** Streamlit Community
Cloud cannot carry them. The deployed app always uses the OpenAI path; the
fine-tuned model is used from the notebook, and its dependencies live in
`requirements-train.txt`.

---
<br></br>

## To-Do List

- [x] SQLite calendar and function calling
- [x] Chroma retrieval over the job description
- [x] Main Agent with baseline and confusion matrix
- [x] Exit Advisor fine-tuning data
- [x] LoRA adapter trained and committed to `models/exit-advisor-lora/`
- [x] Exit Advisor measured before and after fine-tuning
- [x] Final eval numbers in this README — measured 27 November 2026
- [ ] Streamlit Community Cloud deployment URL

---
<br></br>

## License

Coursework, submitted for the Technion School of Continuing Education.

---
<br></br>

## Contact

**Oleg Rogazkin** — HTCGAI4, cohort June 2026

**Mamon Dehabra** — HTCGAI4, cohort June 2026

---
<br></br>

## Acknowledgments

- [LangChain](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs/overview)
- [Chroma](https://docs.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- Amir Ben Haim — course instructor

---
