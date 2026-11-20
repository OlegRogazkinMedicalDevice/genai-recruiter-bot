"""Info Advisor - answers questions about the role from the job description.

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
