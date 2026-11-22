# Deploying to Streamlit Community Cloud

Fifteen minutes if nothing surprises you. The two things that usually do are
already handled in this repository, and both are noted below.

## Before you start

The repository must contain, committed:

- `chroma_store/` — built by `scripts/build_chroma.py`
- `data/schedule.db` — the interview calendar
- `requirements.txt` — **without** torch or transformers

and must **not** contain `.env`. Check with:

```
git ls-files | findstr /i "\.env chroma_store schedule.db"
```

`.env` must not appear in that list. `chroma_store/` and `schedule.db` must.

## 1. Push

```
git remote add origin https://github.com/<your-username>/genai-recruiter-bot.git
git push -u origin main
```

## 2. Create the app

1. `share.streamlit.io` → sign in with GitHub
2. **Create app** → **Deploy a public app from GitHub**
3. Repository: your repo. Branch: `main`.
4. **Main file path**: `streamlit_app/streamlit_main.py`
5. Do not deploy yet — open **Advanced settings** first

## 3. Secrets

In **Advanced settings → Secrets**, paste:

```toml
OPENAI_API_KEY = "sk-..."
CHAT_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MAIN_AGENT_PROMPT_VERSION = "v3"
```

TOML, so the values are in double quotes — unlike `.env`, which has none.

Leave `LOCAL_EXIT_MODEL` out. The fine-tuned model is a local folder with a
torch dependency; the deployed app uses the OpenAI path on purpose.

## 4. Deploy

First build takes a few minutes. Then open the URL and have one conversation
end to end: a question about the role, a scheduling request, a confirmation.

## The two traps, already handled

**sqlite.** chromadb needs sqlite ≥ 3.35 and Streamlit Cloud ships an older
one. Without a fix the app dies the first time the Info Advisor is used.
`streamlit_main.py` swaps in `pysqlite3` before anything imports chromadb, and
`pysqlite3-binary` is in `requirements.txt` for linux only.

**The calendar.** `db_Tech.sql` builds an MS SQL Server database on your own
machine, which the cloud cannot reach. The app reads `data/schedule.db`
instead, which is committed.

## If the build fails

The log is on the app page, under **Manage app**.

| What you see | What it is |
|---|---|
| `ModuleNotFoundError` | the package is missing from `requirements.txt` |
| `sqlite3 version` / `unsupported version` | `pysqlite3-binary` did not install — check it is in `requirements.txt` |
| `FileNotFoundError: chroma_store` | not committed; run the build script and commit it |
| `OPENAI_API_KEY is not set` | Secrets not saved, or saved to a different app |
| app exceeds resource limits | torch or transformers got into `requirements.txt` |

## Last step

Put the URL in `README.md`, under Usage. It is part of what is submitted.
