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
- [x] Retrieval-augmented answers over the job description (Chroma)
- [x] Function calling against a real interview calendar (SQLite)
- [x] Relative dates (“next Friday”) resolved from the conversation's own clock
- [x] Closed days handled explicitly instead of returning nothing
- [x] Fine-tuned Exit Advisor for candidate withdrawals
- [x] Evals with a baseline, a ceiling, and a confusion matrix
- [x] Streamlit interface, deployed to Streamlit Community Cloud

---
<br></br>

## Getting Started

### Prerequisites

- Python >= 3.10
- An OpenAI API key

### Installation

```bash
git clone https://github.com/<your-username>/genai-recruiter-bot.git
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
recruiter turns in `sms_conversations.json`.

| | |
|---|---|
| Eval items | 59 (continue 25 / schedule 19 / end 15) |
| Baseline — always `continue` | **42.37%** |
| Random — 1 of 3 | 33.33% |
| Ceiling — contradictory labels | **96.61%** |
| Main Agent, prompted | _fill in after running the notebook_ |

Two things the notebook establishes before the model is ever called:

- **The baseline.** Answering `continue` every time scores 42.37%. Anything
  that does not beat that number is not working.
- **The ceiling.** Two pairs of identical candidate messages carry different
  labels, so two items are unanswerable by construction. 96.61% is a perfect
  score on this data, not 100%.

The test set is small: one item is 1.7%, so a two-point difference between two
prompts is noise.

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
│   └── build_finetune_data.py      # training + held-out validation JSONL
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

---
<br></br>

## To-Do List

- [x] SQLite calendar and function calling
- [x] Chroma retrieval over the job description
- [x] Main Agent with baseline and confusion matrix
- [x] Exit Advisor fine-tuning data
- [ ] Fine-tuned model id in `.env`
- [ ] Streamlit Community Cloud deployment URL
- [ ] Final eval numbers in this README

---
<br></br>

## License

Coursework, submitted for the Technion School of Continuing Education.

---
<br></br>

## Contact

**Oleg Rogazkin** — HTCGAI4, cohort June 2026

---
<br></br>

## Acknowledgments

- [LangChain](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs/overview)
- [Chroma](https://docs.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- Amir Ben Haim — course instructor

---
