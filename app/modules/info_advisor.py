"""Info Advisor - answers questions about the role from the job description.

The Main Agent does not call answer() directly. It calls consult(), which is
the shape the project brief asks for: the advisor first makes its own decision
- is information actually needed here? - and only then reaches for the vector
store. When it decides information is not needed it hands the turn back to the
Main Agent, which consults another advisor instead.

The PDF is embedded once by scripts/build_chroma.py and persisted to
chroma_store/, which is committed to the repository. Streamlit Community Cloud
then only reads it - no embedding at startup, no PDF parsing in the cloud.
"""

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from . import config

_store: Optional[Chroma] = None


def get_store() -> Chroma:
    """Open the persisted vector store (cached for the process)."""
    global _store
    if _store is None:
        if not config.CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"{config.CHROMA_DIR} not found. Run: python scripts/build_chroma.py")
        _store = Chroma(
            persist_directory=str(config.CHROMA_DIR),
            embedding_function=OpenAIEmbeddings(model=config.EMBEDDING_MODEL,
                                                api_key=config.OPENAI_API_KEY),
            collection_name="job_description")
    return _store


def retrieve(question: str, k: int = 3) -> List[str]:
    """The k most relevant chunks of the job description."""
    return [doc.page_content for doc in get_store().similarity_search(question, k=k)]


def answer(question: str, llm: Optional[ChatOpenAI] = None) -> str:
    """Answer a candidate's question, grounded in the job description."""
    llm = llm or ChatOpenAI(model=config.CHAT_MODEL,
                            temperature=config.TEMPERATURE,
                            api_key=config.OPENAI_API_KEY)
    excerpts = "\n---\n".join(retrieve(question))
    prompt = (f"Job description excerpts:\n{excerpts}\n\n"
              f"Candidate's question: {question}")
    response = llm.invoke([SystemMessage(content=config.INFO_ADVISOR_SYSTEM_PROMPT),
                           HumanMessage(content=prompt)])
    return response.content.strip()


# --- consultation ---------------------------------------------------------

_NEEDS_INFO_PROMPT = (
    "You advise a recruiter's assistant. Decide whether the candidate's last "
    "message needs an answer drawn from the job description - duties, stack, "
    "seniority, location, salary, process.\n"
    "Answer with exactly one word:\n"
    "  yes  - the candidate asked something about the role\n"
    "  no   - the candidate asked nothing, or asked about scheduling only\n"
    "No punctuation, no explanation."
)


def needs_info(turns: List[dict], question: str,
               llm: Optional[ChatOpenAI] = None) -> bool:
    """The advisor's own decision: Info Needed, or Info Not Needed."""
    if not question.strip():
        return False
    llm = llm or ChatOpenAI(model=config.CHAT_MODEL,
                            temperature=config.TEMPERATURE,
                            api_key=config.OPENAI_API_KEY,
                            max_tokens=5)
    history = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns[-6:])
    response = llm.invoke([SystemMessage(content=_NEEDS_INFO_PROMPT),
                           HumanMessage(content=f"Conversation:\n{history}\n\n"
                                                f"Last candidate message: {question}")])
    return response.content.strip().lower().startswith("y")


def consult(turns: List[dict], question: str,
            llm: Optional[ChatOpenAI] = None) -> dict:
    """Advise the Main Agent. Returns a consultation, never a sent message.

    decision is "info_needed" or "info_not_needed". Only the first does the
    vector retrieval; the second is the branch that sends the turn back to the
    Main Agent so another advisor can be consulted.
    """
    if not needs_info(turns, question, llm):
        return {"advisor": "info", "decision": "info_not_needed",
                "draft": "", "evidence": []}

    excerpts = retrieve(question)
    return {"advisor": "info", "decision": "info_needed",
            "draft": answer(question, llm), "evidence": excerpts}
