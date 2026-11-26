"""Main Agent - decides what the recruiter does next.

Output is one of: continue / schedule / end.
This is the component the evals measure, so it is deliberately boring:
temperature 0, one word out, a normalizer that tolerates a chatty model.
"""

import re
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import config


def build_llm(model: Optional[str] = None) -> ChatOpenAI:
    return ChatOpenAI(model=model or config.CHAT_MODEL,
                      temperature=config.TEMPERATURE,
                      api_key=config.OPENAI_API_KEY,
                      max_tokens=5)


def turns_to_messages(turns: List[dict],
                      system_prompt: Optional[str] = None) -> list:
    """Conversation history -> LangChain messages.

    The recruiter is the assistant, the candidate is the human. The turn being
    predicted is NOT included: the model sees everything before it and decides
    what the recruiter should say next.

    system_prompt overrides the configured one, which is how the notebook
    scores several prompt versions against each other in a single run.
    """
    messages = [SystemMessage(content=system_prompt
                              or config.MAIN_AGENT_SYSTEM_PROMPT)]
    for turn in turns:
        if turn["speaker"] == "recruiter":
            messages.append(AIMessage(content=turn["text"]))
        else:
            messages.append(HumanMessage(content=turn["text"]))
    return messages


def normalize(answer: str) -> str:
    """Pull one of the 3 labels out of whatever the model said.

    An answer containing none of them becomes "invalid", which counts as a
    mistake. The first version returned the raw text instead, and a model that
    answered as the recruiter ("Great! Are you available...") then entered the
    results as a brand new label, adding phantom rows and columns to the
    confusion matrix. Credit for this one goes to mamon96d, whose eval had it
    right from the start.

    Word boundaries matter too: without them, "rescheduled" would match
    "schedule".
    """
    text = answer.strip().lower()
    for label in config.LABELS:
        if re.search(rf"\b{label}\b", text):
            return label
    return "invalid"


def decide(turns: List[dict], llm: Optional[ChatOpenAI] = None,
           system_prompt: Optional[str] = None) -> str:
    """Return 'continue' | 'schedule' | 'end' for the next recruiter turn."""
    llm = llm or build_llm()
    messages = turns_to_messages(turns, system_prompt)
    if len(messages) == 1:                      # no history yet: open the screen
        return "continue"
    response = llm.invoke(messages)
    return normalize(response.content)
