import time
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from support_rag import answer_question, BackendError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("support_api")

app = FastAPI(title="Healthcare RAG API", version="1.0")


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty.")
        if len(v) > 1000:
            raise ValueError("Question must be at most 1000 characters.")
        return v


@app.get("/health")
def health():
    # Only shows the API process is up; does not verify Ollama.
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    start = time.perf_counter()
    try:
        answer, sources = answer_question(req.question)
    except BackendError:
        ms = (time.perf_counter() - start) * 1000
        log.error("ask FAILED reason=backend latency_ms=%.1f", ms)
        raise HTTPException(status_code=502, detail="The language model service is currently unavailable.")
    except Exception:
        ms = (time.perf_counter() - start) * 1000
        log.exception("ask FAILED reason=unexpected latency_ms=%.1f", ms)  # trace stays in the log
        raise HTTPException(status_code=500, detail="Internal server error.")

    ms = (time.perf_counter() - start) * 1000
    log.info("ask OK latency_ms=%.1f", ms)  # question text deliberately not logged
    return {"answer": answer, "sources": sources, "response_time_ms": round(ms, 1)}