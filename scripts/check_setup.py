"""Check the project is ready to run, before spending time on it.

    python scripts/check_setup.py

Reports what is in place and what is not. Does not call the OpenAI API except
for one tiny request to confirm the key works, which costs a fraction of a cent.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, MISSING, WARN = "  ok   ", " MISS  ", " warn  "
problems = 0


def report(status, label, detail=""):
    global problems
    if status is MISSING:
        problems += 1
    print(f"[{status}] {label}{'  — ' + detail if detail else ''}")


# --- packages ------------------------------------------------------------
for package in ("openai", "langchain_openai", "langchain_chroma", "chromadb",
                "streamlit", "pandas", "sklearn", "seaborn", "dotenv", "pypdf"):
    try:
        __import__(package)
        report(OK, f"package {package}")
    except ImportError:
        report(MISSING, f"package {package}", "pip install -r requirements.txt")

from app.modules import config  # noqa: E402

# --- key -----------------------------------------------------------------
if not config.OPENAI_API_KEY:
    report(MISSING, "OPENAI_API_KEY", "copy .env.example to .env and fill it in")
else:
    report(OK, "OPENAI_API_KEY", f"{config.OPENAI_API_KEY[:6]}... (not printed in full)")

# --- data ----------------------------------------------------------------
for path, label in ((config.SCHEDULE_DB, "data/schedule.db"),
                    (config.CONVERSATIONS_JSON, "data/sms_conversations.json"),
                    (config.JOB_DESCRIPTION_PDF, "data/job_description.pdf")):
    report(OK if path.exists() else MISSING, label,
           "" if path.exists() else "missing from the repository")

if config.SCHEDULE_DB.exists():
    with sqlite3.connect(config.SCHEDULE_DB) as con:
        rows = con.execute("SELECT COUNT(*) FROM Schedule").fetchone()[0]
        free = con.execute("SELECT COUNT(*) FROM Schedule WHERE available = 1").fetchone()[0]
    report(OK if rows == 9396 else WARN, "schedule rows", f"{rows} rows, {free} free")

# --- built artefacts -----------------------------------------------------
report(OK if config.CHROMA_DIR.exists() else MISSING, "chroma_store/",
       "" if config.CHROMA_DIR.exists() else "run: python scripts/build_chroma.py")

train = config.DATA_DIR / "exit_advisor_train.jsonl"
report(OK if train.exists() else MISSING, "exit_advisor_train.jsonl",
       "" if train.exists() else "run: python scripts/build_finetune_data.py")

report(OK if config.FINETUNED_EXIT_MODEL else WARN, "FINETUNED_EXIT_MODEL",
       config.FINETUNED_EXIT_MODEL or "not set — Exit Advisor falls back to the base model")

# --- secrets hygiene -----------------------------------------------------
gitignore = (Path(config.ROOT) / ".gitignore")
if gitignore.exists() and ".env" in gitignore.read_text():
    report(OK, ".gitignore covers .env")
else:
    report(MISSING, ".gitignore does NOT cover .env", "fix before committing")

# --- live key check ------------------------------------------------------
if config.OPENAI_API_KEY:
    try:
        from langchain_openai import ChatOpenAI
        answer = ChatOpenAI(model=config.CHAT_MODEL, temperature=0, max_tokens=3,
                            api_key=config.OPENAI_API_KEY).invoke("Say ok").content
        report(OK, f"API call to {config.CHAT_MODEL}", repr(answer.strip()))
    except Exception as error:                                   # noqa: BLE001
        report(MISSING, "API call", f"{type(error).__name__}: {error}"[:120])

print()
print("ready" if not problems else f"{problems} thing(s) to fix before starting")
sys.exit(1 if problems else 0)
