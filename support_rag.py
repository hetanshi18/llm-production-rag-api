"""Lightweight RAG pipeline (fits Render's 512 MB free tier).

The index is built locally by build_index.py and loaded from disk here.
Embeddings use fastembed (ONNX) instead of PyTorch; search is plain numpy.
"""
import os
import json
import logging

import numpy as np
import requests
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()
log = logging.getLogger("support_rag")

# ---------- Configuration ----------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

# Distance between normalized vectors (lower = more similar, range 0-2).
# If the best chunk is farther than this, treat the question as unsupported.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "1.1"))

FALLBACK = "I don't have enough information in the provided knowledge base to answer that."


class BackendError(Exception):
    """Raised when the Groq LLM backend fails."""


# ---------- Load once at import ----------
_embedder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
_embeddings = np.load("index_embeddings.npy")
with open("index_chunks.json", encoding="utf-8") as f:
    _chunks = json.load(f)
log.info("Index loaded: %d chunks", len(_chunks))


# ---------- Pipeline ----------
def retrieve(question, k=6):
    """Return list of (chunk_dict, distance), best first."""
    q = np.array(list(_embedder.embed([question]))[0], dtype=np.float32)
    q /= np.linalg.norm(q)
    dists = np.linalg.norm(_embeddings - q, axis=1)
    order = np.argsort(dists)[: k + 8]
    results = []
    for i in order:
        c = _chunks[i]
        # skip front matter (title page / table of contents) of the PDF
        if c["page"] is not None and c["page"] < 6:
            continue
        results.append((c, float(dists[i])))
    return results[:k]


def call_llm(prompt):
    if not GROQ_API_KEY or not GROQ_MODEL:
        log.error("Groq API key or model is not configured.")
        raise BackendError("Groq configuration missing")
    try:
        r = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_completion_tokens": 1024,
                "reasoning_effort": "low",
                "include_reasoning": False,
            },
            timeout=LLM_TIMEOUT,
        )
        r.raise_for_status()
        response = r.json()
        choice = response["choices"][0]
        answer = choice["message"].get("content")
        log.info("Groq finish_reason=%s usage=%s", choice.get("finish_reason"), response.get("usage"))
        if not answer or not answer.strip():
            raise BackendError("Groq returned empty content")
        return answer.strip()
    except BackendError:
        raise
    except Exception as e:
        log.error("Groq backend failure: %s", type(e).__name__)
        raise BackendError(type(e).__name__) from e


def answer_question(question):
    """Returns (answer, sources). Raises BackendError if the LLM fails."""
    results = retrieve(question)
    if results:
        log.info("best_distance=%.3f", results[0][1])

    if not results or results[0][1] > MAX_DISTANCE:
        return FALLBACK, []

    context = "\n\n".join(
        f"[Source: {c['source']}, Page: {c['page'] if c['page'] is not None else 'unknown'}]\n{c['text']}"
        for c, _ in results
    )

    prompt = f"""
You are a cautious educational healthcare information assistant.

Answer the question only using the supplied context.

If the context only partly answers the question, give what it does say and note what is missing.
Only say the information is not available if the context has nothing relevant.

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
        f"{c['source']} (page {c['page'] if c['page'] is not None else 'n/a'})"
        for c, _ in results
    })
    return answer, sources