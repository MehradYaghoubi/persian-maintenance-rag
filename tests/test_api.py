from fastapi.testclient import TestClient

from app import create_app
from retrieval import Engine, normalize


def test_normalization():
    assert normalize("ياد‌آوري كد e۱۷، نسخه ۲٫۰") == "یاد آوری کد E17، نسخه 2.0"


def test_exact_code_is_scoped_to_the_manual():
    client = TestClient(create_app())
    answers = [
        client.post("/ask", json={"question": "کد e۱۷ یعنی چی؟", "model": model, "revision": "2.0"}).json()
        for model in ("CN-120", "CN-240")
    ]
    assert [item["status"] for item in answers] == ["answered", "answered"]
    assert [item["citations"][0]["id"] for item in answers] == ["CN-120/2.0/E17", "CN-240/2.0/E17"]
    assert answers[0]["answer"] != answers[1]["answer"]
    assert all(item["answer"] == item["citations"][0]["excerpt"] for item in answers)


def test_missing_metadata_only_blocks_differing_or_missing_rows():
    client = TestClient(create_app())
    ambiguous = client.post("/ask", json={"question": "E17"}).json()
    assert ambiguous["status"] == "clarify"
    assert ambiguous["missing"] == ["model", "revision"]
    assert not ambiguous["citations"]
    ambiguous = client.post("/ask", json={"question": "E12", "model": "CN-120"}).json()
    assert ambiguous["missing"] == ["revision"]
    stable = client.post("/ask", json={"question": "E04", "model": "CN-240"}).json()
    assert {item["revision"] for item in stable["citations"]} == {"1.0", "2.0"}
    absent = client.post("/ask", json={"question": "E12", "model": "CN-120", "revision": "1.0"}).json()
    assert absent["status"] == "abstain"
    assert absent["human_review"]


def test_metadata_in_question_and_conflict():
    client = TestClient(create_app())
    question = "در cn ۲۴۰ ویرایش ۱٫۰ خطای E۱۷ یعنی چه؟"
    assert client.post("/ask", json={"question": question}).json()["citations"][0]["id"] == "CN-240/1.0/E17"
    conflict = client.post("/ask", json={"question": question, "model": "CN-120"}).json()
    assert conflict["missing"] == ["model"]


def test_literal_precedence_and_hybrid_paraphrase():
    engine = Engine()
    assert engine.ask("E99 و گزارش دما", "CN-120", "1.0")[0]["status"] == "abstain"
    assert engine.ask("E17 یا E31", "CN-120", "1.0")[0]["status"] == "abstain"
    spelling = "ورژن فریمور را کجا پیدا کنم؟"
    assert engine.ask(spelling, "CN-120", "2.0", mode="lexical")[0]["status"] == "abstain"
    assert engine.ask(spelling, "CN-120", "2.0")[0]["citations"][0]["id"] == "CN-120/2.0/revision"


def test_unanswerable_and_validation():
    client = TestClient(create_app())
    answer = client.post("/ask", json={"question": "شماره تلفن خدمات مرکزی چیست؟"}).json()
    assert answer["status"] == "abstain"
    assert answer["human_review"]
    assert answer["citations"] == []
    for body in ({"question": "  "}, {"question": 42}, {"question": "E17", "model": "CN-999"}):
        assert client.post("/ask", json=body).status_code == 422
    assert client.get("/health").json() == {"status": "ok", "retrieval": "hybrid-rules"}
    assert 'dir="rtl"' in client.get("/").text
