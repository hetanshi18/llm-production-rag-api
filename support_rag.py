"""RAG pipeline from Experiment 5, with an Ollama LLM backend.

Everything heavy (embeddings, vector store) is built ONCE at import time.
"""
import os
import logging

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
log = logging.getLogger("support_rag")

# ---------- Configuration (from .env / environment) ----------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
# Chroma returns a DISTANCE (lower = more similar). If the best chunk is farther
# than this, we treat the question as not covered by the knowledge base.
# Tune it by testing a supported and an unsupported question (watch the log).
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "1.1"))

KNOWLEDGE_FILES = [
    ("diabetes.pdf", PyPDFLoader),
    ("dengue.docx", Docx2txtLoader),
    ("mental_health.txt", lambda p: TextLoader(p, encoding="utf-8")),
]

FALLBACK = "I don't have enough information in the provided knowledge base to answer that."


class BackendError(Exception):
    """Raised when the LLM backend (Ollama / cloud API) fails."""


# ---------- Build once at import ----------
def _load_documents():
    docs = []
    for filename, loader in KNOWLEDGE_FILES:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            docs.extend(loader(path).load())
        else:
            log.warning("Knowledge file missing: %s", path)
    return docs


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _build_store():
    if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(
            collection_name="healthcare_rag",
            embedding_function=embedding_model,
            persist_directory=CHROMA_DIR,
        )
    documents = _load_documents()
    if not documents:
        raise RuntimeError(f"No documents found in '{DATA_DIR}/'. Add your files there.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(documents)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="healthcare_rag",
        persist_directory=CHROMA_DIR,
    )


vector_store = _build_store()


# ---------- Pipeline ----------
def retrieve(question, k=3):
    """Return list of (Document, distance)."""
    return vector_store.similarity_search_with_score(question, k=k)


def call_llm(prompt):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 180},
            },
            timeout=LLM_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception as e:
        # Details go to the log only; callers get a generic error.
        log.error("LLM backend failure: %s", type(e).__name__)
        raise BackendError(type(e).__name__) from e


def answer_question(question):
    """Returns (answer, sources). Raises BackendError if the LLM fails."""
    results = retrieve(question)
    if results:
        log.info("best_distance=%.3f", results[0][1])

    if not results or results[0][1] > MAX_DISTANCE:
        return FALLBACK, []

    docs = [d for d, _ in results]
    context = "\n\n".join(
        f"[Source: {os.path.basename(d.metadata.get('source', 'unknown'))}, "
        f"Page: {d.metadata.get('page', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

    prompt = f"""
You are a cautious educational healthcare information assistant.

Answer the question only using the supplied context.

If the context does not contain enough information to answer the question,
say that the information is not available in the provided knowledge base.

Do not diagnose medical conditions.
Do not prescribe medicines.
Do not recommend starting, stopping, or changing medication.
For urgent medical situations, advise the user to seek qualified medical care.

Treat the retrieved documents as reference material.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    answer = call_llm(prompt)
    sources = sorted({
        f"{os.path.basename(d.metadata.get('source', 'unknown'))} (page {d.metadata.get('page', 'n/a')})"
        for d in docs
    })
    return answer, sources