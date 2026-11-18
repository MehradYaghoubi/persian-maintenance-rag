import logging
import os
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, StringConstraints

from retrieval import ROOT, Engine
from semantic import DependencyError, Semantic

log = logging.getLogger(__name__)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=500)]
    model: Literal["CN-120", "CN-240"] | None = None
    revision: Literal["1.0", "2.0"] | None = None


class Citation(BaseModel):
    id: str
    title: str
    model: Literal["CN-120", "CN-240"]
    revision: Literal["1.0", "2.0"]
    section: str
    code: str | None
    excerpt: str


class Answer(BaseModel):
    status: Literal["answered", "clarify", "abstain"]
    answer: str
    citations: list[Citation]
    missing: list[Literal["model", "revision"]]
    human_review: bool


def create_app(engine=None) -> FastAPI:
    if engine is None:
        engine = Engine(Semantic() if os.getenv("DATABASE_URL") or os.getenv("OLLAMA_URL") else None)
    api = FastAPI(title="Persian Maintenance RAG", version="1.0.0")
    unavailable = {503: {"description": "The retrieval service is temporarily unavailable"}}

    @api.get("/", response_class=FileResponse, include_in_schema=False)
    def home():
        return FileResponse(ROOT / "demo.html", media_type="text/html; charset=utf-8")

    @api.get("/health", responses=unavailable)
    def health() -> dict[str, str]:
        if engine.semantic:
            try:
                engine.semantic.check()
            except DependencyError:
                log.exception("Retrieval health check failed")
                raise HTTPException(503, "Retrieval service is temporarily unavailable")
        return {"status": "ok", "retrieval": engine.retrieval}

    @api.post("/ask", response_model=Answer, responses=unavailable)
    def ask(body: Question):
        try:
            answer, _ = engine.ask(body.question, body.model, body.revision)
        except DependencyError:
            log.exception("Retrieval dependency failed")
            raise HTTPException(503, "Retrieval service is temporarily unavailable")
        return answer

    return api


app = create_app()
