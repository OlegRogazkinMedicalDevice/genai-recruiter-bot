"""Central configuration: paths, models, prompts.

Everything that two modules might disagree about lives here once.
The system prompt in particular: the eval, the fine-tuning file and the
running agent must all use the SAME text, or the measurement is meaningless.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SCHEDULE_DB = DATA_DIR / "schedule.db"
JOB_DESCRIPTION_PDF = DATA_DIR / "job_description.pdf"
CONVERSATIONS_JSON = DATA_DIR / "sms_conversations.json"
CHROMA_DIR = ROOT / "chroma_store"

# --- credentials ---------------------------------------------------------
# Streamlit Community Cloud injects st.secrets; locally we read .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- models --------------------------------------------------------------
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# The Exit Advisor can run in three ways, checked in this order:
#
#   LOCAL_EXIT_MODEL     a folder produced by scripts/finetune_exit_advisor.py,
#                        e.g. models/exit-advisor-lora. This is the route in
#                        use: OpenAI fine-tuning is closed to new accounts.
#   FINETUNED_EXIT_MODEL an OpenAI fine-tuned model id, kept for completeness.
#   (neither)            CHAT_MODEL with a prompt - the "before" in the
#                        before/after comparison.
#
# The local model needs torch and transformers, which are NOT in
# requirements.txt on purpose: Streamlit Community Cloud cannot carry them.
# The deployed app runs the OpenAI path; the local model is for the notebook.
LOCAL_EXIT_MODEL = os.getenv("LOCAL_EXIT_MODEL", "")
FINETUNED_EXIT_MODEL = os.getenv("FINETUNED_EXIT_MODEL", "")

HF_TOKEN = os.getenv("HF_TOKEN", "")

# Classification must be reproducible or the evals move between runs.
TEMPERATURE = 0.0

# --- the position this bot recruits for ----------------------------------
POSITION = "Python Dev"

# --- labels --------------------------------------------------------------
LABELS = ["continue", "schedule", "end"]

# --- prompts -------------------------------------------------------------
# NOTE: MAIN_AGENT_SYSTEM_PROMPT is the string that must also appear in the
# fine-tuning file and in tests/test_evals.ipynb. Change it in one place only.
# --- Main Agent prompt variants ------------------------------------------
# Three versions, so the notebook can score them against each other in one
# run. MAIN_AGENT_PROMPT_VERSION in .env picks the one the app uses.
#
# v1 is the obvious reading of the three labels.
# v2 adds the decision rules that the labelled data actually follows - these
#    were derived by building a rule-based classifier on the 59 items and
#    reading its errors (see app/modules/rule_reference.py, 88.14%).
# v3 is v2 plus few-shot examples. The examples are PARAPHRASED, never copied
#    from the eval set, so the measurement stays honest.

MAIN_AGENT_PROMPT_V1 = (
    "You are the main agent of an SMS recruiting assistant for a Python Developer "
    "role. You read the conversation so far and decide what the recruiter should do "
    "next. Answer with exactly one word:\n"
    "continue - keep screening the candidate, ask about experience or skills, or "
    "answer a question about the job.\n"
    "schedule - the candidate is interested and it is time to offer or negotiate an "
    "interview slot.\n"
    "end - the conversation is finished: either the interview is confirmed, or the "
    "candidate has declined or is not a fit.\n"
    "Respond with only the word."
)

_V2_RULES = (
    "You are the main agent of an SMS recruiting assistant for a Python Developer "
    "role. You read the conversation so far and decide what the recruiter should do "
    "next. Answer with exactly one word: continue, schedule, or end.\n"
    "\n"
    "Apply these rules in order, and stop at the first one that fits.\n"
    "\n"
    "1. The candidate withdraws, says they are not interested, has taken another "
    "offer, or asks to be removed -> end\n"
    "2. The candidate names or accepts a specific day and time, and asks nothing "
    "else -> end. This holds even when that time was not one of the slots you "
    "offered: the recruiter confirms it rather than negotiating again.\n"
    "3. The candidate asks a question, even while accepting a time -> continue. "
    "An unanswered question always outranks the time.\n"
    "4. The candidate says the offered time does not work, that they are busy, or "
    "asks for other times -> schedule. Declining a slot is not declining the job.\n"
    "5. The candidate has just given their first real answer about their Python "
    "experience -> schedule. Screening is over; offer an interview.\n"
    "6. The candidate says they are ready to meet or to book something -> schedule\n"
    "7. Nothing has been said yet, or the candidate is still answering screening "
    "questions -> continue\n"
    "\n"
    "Respond with only the word."
)

_V3_SHOTS = (
    "\n"
    "Examples:\n"
    "\n"
    "candidate: \"Four years with Python, mostly backend APIs.\"\n"
    "answer: schedule\n"
    "\n"
    "candidate: \"None of those slots suit me, I'm teaching that morning.\"\n"
    "answer: schedule\n"
    "\n"
    "candidate: \"Wednesday at 11 is fine.\"\n"
    "answer: end\n"
    "\n"
    "candidate: \"Wednesday at 11 is fine. Is the role hybrid or on site?\"\n"
    "answer: continue\n"
    "\n"
    "candidate: \"I've signed with another company, please close my file.\"\n"
    "answer: end\n"
    "\n"
    "candidate: \"What does the team work on day to day?\"\n"
    "answer: continue\n"
)

MAIN_AGENT_PROMPT_V2 = _V2_RULES
MAIN_AGENT_PROMPT_V3 = _V2_RULES.replace(
    "\nRespond with only the word.", _V3_SHOTS + "\nRespond with only the word.")

MAIN_AGENT_PROMPTS = {"v1": MAIN_AGENT_PROMPT_V1,
                      "v2": MAIN_AGENT_PROMPT_V2,
                      "v3": MAIN_AGENT_PROMPT_V3}

MAIN_AGENT_PROMPT_VERSION = os.getenv("MAIN_AGENT_PROMPT_VERSION", "v3")
MAIN_AGENT_SYSTEM_PROMPT = MAIN_AGENT_PROMPTS[MAIN_AGENT_PROMPT_VERSION]

INFO_ADVISOR_SYSTEM_PROMPT = (
    "You are the info advisor of an SMS recruiting assistant. Answer the candidate's "
    "question about the Python Developer role using only the job description excerpts "
    "provided. If the excerpts do not contain the answer, say you will check with the "
    "hiring manager. Keep the answer under 300 characters - this is an SMS."
)

SCHEDULING_ADVISOR_SYSTEM_PROMPT = (
    "You are the scheduling advisor of an SMS recruiting assistant. You are given the "
    "interview slots that are actually free in the company calendar. Offer them to the "
    "candidate in a short, friendly SMS. Never invent a slot that is not in the list. "
    "If the candidate asked for a time that is not available, say so plainly and offer "
    "the closest alternatives from the list. Keep it under 300 characters."
)

EXIT_ADVISOR_SYSTEM_PROMPT = (
    "You are the exit advisor of an SMS recruiting assistant. The conversation is "
    "ending. Write the final SMS. If the interview was confirmed, confirm it warmly "
    "and say a calendar invite follows. If the candidate declined or withdrew, thank "
    "them, leave the door open, and do not try to change their mind. "
    "Keep it under 300 characters."
)
