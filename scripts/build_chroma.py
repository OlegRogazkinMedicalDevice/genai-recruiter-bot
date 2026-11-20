"""Embed the job description into Chroma, once, offline.

Run this on your own machine and commit chroma_store/ to the repository.
Streamlit Community Cloud then only reads it: no PDF parsing and no embedding
calls at startup, which keeps the deployed app fast and cheap.

    python scripts/build_chroma.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pypdf                                                         # noqa: E402
from langchain_chroma import Chroma                                  # noqa: E402
from langchain_core.documents import Document                        # noqa: E402
from langchain_openai import OpenAIEmbeddings                        # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from app.modules import config                                       # noqa: E402

if not config.OPENAI_API_KEY:
    sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")

if not config.JOB_DESCRIPTION_PDF.exists():
    sys.exit(f"{config.JOB_DESCRIPTION_PDF} not found.")

print(f"loading  {config.JOB_DESCRIPTION_PDF.name}")
reader = pypdf.PdfReader(str(config.JOB_DESCRIPTION_PDF))
pages = [Document(page_content=page.extract_text() or "",
                  metadata={"source": config.JOB_DESCRIPTION_PDF.name, "page": number})
         for number, page in enumerate(reader.pages, start=1)]
print(f"         {len(pages)} pages")

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = splitter.split_documents(pages)
print(f"split    {len(chunks)} chunks")

if config.CHROMA_DIR.exists():
    shutil.rmtree(config.CHROMA_DIR)

store = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model=config.EMBEDDING_MODEL,
                               api_key=config.OPENAI_API_KEY),
    persist_directory=str(config.CHROMA_DIR),
    collection_name="job_description")

print(f"stored   {config.CHROMA_DIR}")

# sanity check: a question whose answer is in the PDF
hits = store.similarity_search("Which Python frameworks are preferred?", k=2)
print("\ncheck    'Which Python frameworks are preferred?'")
for hit in hits:
    print("  ->", hit.page_content[:120].replace("\n", " "), "...")
