"""Exit Advisor - writes the last SMS of the conversation.

Two very different endings share the label 'end':
  - the interview is confirmed  (11 of the 15 conversations)
  - the candidate declined      ( 4 of the 15: conversations 5, 8, 13, 15)

A prompted base model tends to treat both as a happy ending and congratulates
a candidate who just withdrew. That is the specific failure fine-tuning is
meant to fix, which is why the training file is built from the four refusals
plus confirmations as counter-examples (scripts/build_finetune_data.py).

Three backends, picked by what is set in .env - see config.py. The local one
is loaded lazily so that neither torch nor transformers is needed to run the
deployed app.
"""

from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import config

_local = None          # (tokenizer, model) once loaded


def backend_name() -> str:
    """Which backend will answer - useful to print in the eval notebook."""
    if config.LOCAL_EXIT_MODEL:
        return f"local:{config.LOCAL_EXIT_MODEL}"
    if config.FINETUNED_EXIT_MODEL:
        return f"openai:{config.FINETUNED_EXIT_MODEL}"
    return f"openai:{config.CHAT_MODEL} (not fine-tuned)"


def to_chat_messages(turns: List[dict]) -> List[dict]:
    """Conversation turns -> plain role/content dicts."""
    return [{"role": "assistant" if turn["speaker"] == "recruiter" else "user",
             "content": turn["text"]}
            for turn in turns[-8:]]


# --- local (Hugging Face) -------------------------------------------------

def _load_local():
    """Load the fine-tuned adapter once per process."""
    global _local
    if _local is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = config.LOCAL_EXIT_MODEL
        tokenizer = AutoTokenizer.from_pretrained(path)

        try:                                    # a LoRA adapter folder
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        except Exception:                       # a fully fine-tuned model
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)

        model.eval()
        _local = (tokenizer, model)
    return _local


def _close_local(turns: List[dict]) -> str:
    import torch

    tokenizer, model = _load_local()
    messages = ([{"role": "system", "content": config.EXIT_ADVISOR_SYSTEM_PROMPT}]
                + to_chat_messages(turns))
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():                       # greedy: reproducible evals
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id
                                or tokenizer.eos_token_id)

    reply = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(reply, skip_special_tokens=True).strip()


# --- OpenAI ---------------------------------------------------------------

def build_llm(model: Optional[str] = None) -> ChatOpenAI:
    return ChatOpenAI(model=model or config.FINETUNED_EXIT_MODEL or config.CHAT_MODEL,
                      temperature=config.TEMPERATURE,
                      api_key=config.OPENAI_API_KEY)


def _close_openai(turns: List[dict], llm: Optional[ChatOpenAI]) -> str:
    llm = llm or build_llm()
    messages = [SystemMessage(content=config.EXIT_ADVISOR_SYSTEM_PROMPT)]
    for turn in turns[-8:]:
        if turn["speaker"] == "recruiter":
            messages.append(AIMessage(content=turn["text"]))
        else:
            messages.append(HumanMessage(content=turn["text"]))
    return llm.invoke(messages).content.strip()


# --- entry point ----------------------------------------------------------

def close(turns: List[dict], llm: Optional[ChatOpenAI] = None,
          force_base: bool = False) -> str:
    """Write the closing SMS.

    force_base=True always uses the un-tuned OpenAI model, which is how the
    notebook produces the "before" side of the comparison in one pass.
    """
    if force_base:
        return _close_openai(turns, ChatOpenAI(model=config.CHAT_MODEL,
                                               temperature=config.TEMPERATURE,
                                               api_key=config.OPENAI_API_KEY))
    if config.LOCAL_EXIT_MODEL:
        return _close_local(turns)
    return _close_openai(turns, llm)


# --- consultation ---------------------------------------------------------

_SHOULD_END_PROMPT = (
    "You advise a recruiter's assistant. Decide whether this conversation "
    "should be closed now.\n"
    "Answer with exactly one word:\n"
    "  yes  - the candidate withdrew, or a time is agreed and nothing is open\n"
    "  no   - something is still unresolved and the conversation continues\n"
    "No punctuation, no explanation."
)


def should_end(turns: List[dict], llm: Optional[ChatOpenAI] = None) -> bool:
    """The advisor's own decision: End Conversation, or Don't End Conv."""
    llm = llm or ChatOpenAI(model=config.CHAT_MODEL,
                            temperature=config.TEMPERATURE,
                            api_key=config.OPENAI_API_KEY,
                            max_tokens=5)
    history = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns[-6:])
    response = llm.invoke([SystemMessage(content=_SHOULD_END_PROMPT),
                           HumanMessage(content=f"Conversation:\n{history}")])
    return response.content.strip().lower().startswith("y")


def consult(turns: List[dict], llm: Optional[ChatOpenAI] = None) -> dict:
    """Advise the Main Agent.

    The draft on the End branch comes from the fine-tuned model. The Main Agent
    relays it unchanged - paraphrasing it would undo the fine-tuning, which is
    the one thing this advisor exists to demonstrate.
    """
    if not should_end(turns, llm):
        return {"advisor": "exit", "decision": "dont_end",
                "draft": "", "evidence": []}

    return {"advisor": "exit", "decision": "end_conversation",
            "draft": close(turns, llm), "evidence": [backend_name()]}
