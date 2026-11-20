"""Exit Advisor - writes the last SMS of the conversation.

Two very different endings share the label 'end':
  - the interview is confirmed  (11 of the 15 conversations)
  - the candidate declined      ( 4 of the 15: conversations 5, 8, 13, 15)

A prompted base model tends to treat both as a happy ending and congratulates a
candidate who just withdrew. That is the specific failure fine-tuning is meant
to fix here, which is why the training file is built from the four refusals plus
confirmations as counter-examples (scripts/build_finetune_data.py).

Set FINETUNED_EXIT_MODEL in .env once training is done; until then this module
runs the base model with the same system prompt, so the before/after comparison
is like for like.
"""

from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import config


def build_llm(model: Optional[str] = None) -> ChatOpenAI:
    """Use the fine-tuned model when one is configured, else the base model."""
    return ChatOpenAI(model=model or config.FINETUNED_EXIT_MODEL or config.CHAT_MODEL,
                      temperature=config.TEMPERATURE,
                      api_key=config.OPENAI_API_KEY)


def close(turns: List[dict], llm: Optional[ChatOpenAI] = None) -> str:
    llm = llm or build_llm()
    messages = [SystemMessage(content=config.EXIT_ADVISOR_SYSTEM_PROMPT)]
    for turn in turns[-8:]:
        if turn["speaker"] == "recruiter":
            messages.append(AIMessage(content=turn["text"]))
        else:
            messages.append(HumanMessage(content=turn["text"]))
    return llm.invoke(messages).content.strip()
