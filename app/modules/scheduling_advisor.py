"""Scheduling Advisor - offers interview slots that actually exist.

The LLM never invents a time. It is handed the rows the database returned and
writes the SMS around them. The lookup itself is exposed as a LangChain tool,
so the same function is callable by an agent or directly.
"""

import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from . import config, db


@tool
def lookup_interview_slots(after: str, on_date: str = "", limit: int = 3) -> str:
    """Return interview slots that are free in the company calendar.

    after:   ISO timestamp; only slots strictly later than this are returned.
    on_date: optional 'YYYY-MM-DD' to look at a single day.
    limit:   how many slots to return, soonest first.
    """
    if on_date and db.day_is_closed(on_date):
        weekday = datetime.date.fromisoformat(on_date).strftime("%A")
        return (f"The office does not hold interviews on {weekday}s. "
                f"Nearest alternatives: "
                f"{db.describe_slots(db.find_available_slots(after, limit=limit))}")
    slots = db.find_available_slots(after, on_date=on_date or None, limit=limit)
    return db.describe_slots(slots)


def build_llm(model: Optional[str] = None) -> ChatOpenAI:
    return ChatOpenAI(model=model or config.CHAT_MODEL,
                      temperature=config.TEMPERATURE,
                      api_key=config.OPENAI_API_KEY)


def propose(turns: List[dict],
            conversation_start: str,
            requested_phrase: str = "",
            llm: Optional[ChatOpenAI] = None) -> str:
    """Write the recruiter's scheduling SMS.

    requested_phrase is the candidate's own wording ("next Friday", "Monday at
    3"), used to resolve a date against the conversation's own start time.
    """
    llm = llm or build_llm()

    on_date = ""
    note = ""
    if requested_phrase:
        resolved = db.resolve_relative_date(requested_phrase, conversation_start)
        if resolved:
            if db.day_is_closed(resolved):
                note = (f"The candidate asked for {resolved}, which is a day the "
                        f"office does not interview on. Say so and offer the "
                        f"alternatives below.")
            else:
                on_date = resolved

    slots = db.find_available_slots(conversation_start, on_date=on_date or None)
    if not slots:                                # requested day is full
        slots = db.find_available_slots(conversation_start)
        if on_date:
            note = (f"Nothing is free on {on_date}. Offer the alternatives below "
                    f"instead.")

    history = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns[-6:])
    prompt = (f"Conversation so far:\n{history}\n\n"
              f"Free slots: {db.describe_slots(slots)}\n"
              f"{note}\n\nWrite the recruiter's next SMS.")

    response = llm.invoke([SystemMessage(content=config.SCHEDULING_ADVISOR_SYSTEM_PROMPT),
                           HumanMessage(content=prompt)])
    return response.content.strip()
