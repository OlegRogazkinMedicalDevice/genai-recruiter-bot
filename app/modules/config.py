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
MAIN_AGENT_SYSTEM_PROMPT = (
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
