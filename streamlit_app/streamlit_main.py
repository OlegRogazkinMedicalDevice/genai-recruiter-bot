"""Streamlit front end — the SMS conversation, in a browser.

Run locally:   streamlit run streamlit_app/streamlit_main.py
Deployed:      Streamlit Community Cloud, with OPENAI_API_KEY in Secrets.

The key is read from st.secrets in the cloud and from .env locally; .env is
never committed, and the cloud never sees it.
"""

import os
import sys
from pathlib import Path

# Streamlit Community Cloud ships an old system sqlite3, and chromadb requires
# >= 3.35. pysqlite3-binary provides a newer one; this swap must happen before
# anything imports chromadb. Harmless locally, where pysqlite3 is absent.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Secrets must reach the modules before they read config.
for name in ("OPENAI_API_KEY", "CHAT_MODEL", "EMBEDDING_MODEL",
             "MAIN_AGENT_PROMPT_VERSION", "FINETUNED_EXIT_MODEL"):
    try:
        if name in st.secrets:
            os.environ[name] = str(st.secrets[name])
    except Exception:                      # no secrets file locally - fine
        pass

from app.main import RecruiterBot          # noqa: E402
from app.modules import config             # noqa: E402

st.set_page_config(page_title="Python Developer — SMS screening",
                   page_icon="💬", layout="centered")

ACTION_BADGE = {"continue": "🟦 continue", "schedule": "🟩 schedule", "end": "🟥 end"}

# --- sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    st.caption("The calendar covers 2024 only, so the conversation starts there. "
               "Relative dates such as “next Friday” are resolved from this moment, "
               "not from today.")
    start = st.text_input("Conversation start (UTC)", "2024-04-03T15:12:00Z")
    st.divider()
    st.write("**Model**", config.CHAT_MODEL)
    st.write("**Exit advisor**",
             config.FINETUNED_EXIT_MODEL or f"{config.CHAT_MODEL} (not fine-tuned)")
    st.write("**Position**", config.POSITION)
    if st.button("Restart conversation", use_container_width=True):
        st.session_state.clear()
        st.rerun()

if not config.OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Locally: copy .env.example to .env. "
             "On Streamlit Cloud: Settings → Secrets.")
    st.stop()

# --- state ---------------------------------------------------------------
if "bot" not in st.session_state:
    st.session_state.bot = RecruiterBot(conversation_start=start)
    st.session_state.messages = []
    opening = st.session_state.bot.opening()
    st.session_state.messages.append(("assistant", opening["text"], opening["action"]))

st.title("Python Developer — SMS screening")
st.caption("Main Agent routes every message to the Info, Scheduling or Exit advisor.")

for role, text, action in st.session_state.messages:
    with st.chat_message(role):
        st.write(text)
        if action:
            st.caption(ACTION_BADGE.get(action, action))

finished = any(action == "end" for _, _, action in st.session_state.messages[1:])

if finished:
    st.info("This conversation has ended. Use **Restart conversation** in the sidebar.")
else:
    candidate_text = st.chat_input("Reply as the candidate…")
    if candidate_text:
        st.session_state.messages.append(("user", candidate_text, None))
        with st.chat_message("user"):
            st.write(candidate_text)
        with st.chat_message("assistant"):
            with st.spinner("…"):
                reply = st.session_state.bot.respond(candidate_text)
            st.write(reply["text"])
            st.caption(ACTION_BADGE.get(reply["action"], reply["action"]))
        st.session_state.messages.append(("assistant", reply["text"], reply["action"]))
        st.rerun()
