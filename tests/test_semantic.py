import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from app import create_app
from evaluate import load_cases
from retrieval import Engine
from semantic import Semantic


@pytest.fixture
def ollama(monkeypatch):
    requests = []
    state = {"id": "CN-240/2.0/revision"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, data):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def do_GET(self):
            assert self.path == "/api/tags"
            self.send_json({"models": [{"name": "embeddinggemma:latest"}, {"name": "qwen2.5:1.5b"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if self.path == "/api/embed":
                assert body["model"] == "embeddinggemma"
                self.send_json({"embeddings": [
                    [1.0, 0.0] if text.startswith("task:") or "نمایش نسخه" in text else [0.0, 1.0]
                    for text in body["input"]
                ]})
            elif self.path == "/api/generate":
                assert body["stream"] is False
                self.send_json({"response": json.dumps(state)})
            else:
                self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        monkeypatch.setenv("OLLAMA_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        yield requests, state
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_ollama_http_contract_and_allowed_citations(ollama):
    requests, state = ollama
    api = TestClient(create_app())
    assert api.get("/health").json()["retrieval"] == "hybrid-ollama"
    question = {"question": "ورژن فریمور کجاست؟", "model": "CN-240", "revision": "2.0"}
    response = api.post("/ask", json=question).json()
    assert response["status"] == "answered"
    assert response["citations"][0]["id"] == state["id"]
    assert response["answer"] == response["citations"][0]["excerpt"]
    assert requests[0][1]["input"][0].startswith("task: search result | query:")
    assert len(requests[1][1]["input"]) == 32
    assert requests[2][1]["format"]["properties"]["id"]["enum"] == [state["id"], None]
    prompt = json.loads(requests[2][1]["prompt"])
    assert prompt["question"] == question["question"]
    assert list(prompt["sources"]) == [state["id"]]

    state["id"] = "CN-120/1.0/revision"
    assert api.post("/ask", json=question).json()["status"] == "abstain"
    assert [route for route, _ in requests].count("/api/embed") == 3

    calls = len(requests)
    response = api.post("/ask", json={**question, "question": "E17"}).json()
    assert response["citations"][0]["id"] == "CN-240/2.0/E17"
    assert len(requests) == calls


def test_dependency_failure_is_reported(monkeypatch):
    error = RuntimeError("deterministic closed connection")

    def unavailable(*args, **kwargs):
        raise OSError(str(error))

    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr("semantic.urlopen", unavailable)
    api = TestClient(create_app())
    assert api.get("/health").status_code == 503
    response = api.post("/ask", json={"question": "ورژن فریمور کجاست؟"})
    assert response.status_code == 503
    assert str(error) not in response.text


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="requires a PostgreSQL/pgvector test service")
def test_pgvector_seed_and_scope(ollama, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    backend = Semantic()
    backend.seed()
    backend.seed()
    api = TestClient(create_app())
    assert api.get("/health").json()["retrieval"] == "hybrid-pgvector"
    scores = backend.similar("ورژن فریمور کجاست؟", "CN-240", "2.0")
    assert len(scores) == 8
    assert scores["CN-240/2.0/revision"] == pytest.approx(1)
    assert all(key.startswith("CN-240/2.0/") for key in scores)
    response = api.post("/ask", json={
        "question": "ورژن فریمور کجاست؟", "model": "CN-240", "revision": "2.0",
    }).json()
    assert response["citations"][0]["id"] == "CN-240/2.0/revision"


def test_benchmark_labels_reference_unique_queries_and_real_sources():
    dev, test = load_cases("dev"), load_cases("test")
    ids = {row.id for row in Engine().corpus}
    assert (len(dev), len(test)) == (15, 45)
    assert len({case["id"] for case in dev + test}) == 60
    assert len({case["question"] for case in dev + test}) == 60
    for case in dev + test:
        expected = set(filter(None, case["expected"].split(",")))
        if case["status"] == "answered":
            assert expected and expected <= ids
        elif case["status"] == "clarify":
            assert expected and expected <= {"model", "revision"}
        else:
            assert case["status"] == "abstain" and not expected
