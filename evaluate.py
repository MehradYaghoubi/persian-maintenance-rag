import argparse
import csv
import json
import math
import platform
from time import perf_counter

from retrieval import ROOT, Engine
from semantic import Semantic


def load_cases(split):
    with (ROOT / "evaluation" / f"{split}.psv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="|"))


def evaluate(engine, cases, mode):
    timings = []
    recalls = []
    citations = []
    scopes = []
    unsupported = 0
    no_evidence_cases = 0
    correct_abstentions = 0
    expected_abstentions = 0
    correct_status = 0
    failures = []
    for case in cases:
        start = perf_counter()
        response, ranked = engine.ask(case["question"], case["model"] or None, case["revision"] or None, mode=mode)
        timings.append((perf_counter() - start) * 1000)
        status = case["status"]
        predicted = response["status"]
        expected = set(filter(None, case["expected"].split(",")))
        actual = {citation["id"] for citation in response["citations"]}
        matches = status == predicted
        if status == "answered":
            recalls.append(len(expected & {row.id for row in ranked[:5]}) / len(expected))
        if status == "abstain":
            expected_abstentions += 1
            correct_abstentions += predicted == "abstain"
        if predicted == "answered":
            exact_citation = status == "answered" and actual == expected
            valid_scope = status == "answered" and {item.rsplit("/", 1)[0] for item in actual} == {
                item.rsplit("/", 1)[0] for item in expected
            }
            citations.append(exact_citation)
            scopes.append(valid_scope)
            unsupported += not exact_citation
            no_evidence_cases += status != "answered"
            matches = exact_citation
        elif status == "clarify":
            matches = matches and set(response["missing"]) == expected
        correct_status += status == predicted
        if not matches:
            failures.append({
                "id": case["id"], "expected_status": status, "status": predicted,
                "expected": sorted(expected), "actual": sorted(actual or set(response["missing"])),
            })

    def mean(items):
        return round(sum(items) / len(items), 4) if items else None

    return {
        "queries": len(cases),
        "expected_answered": len(recalls),
        "expected_abstain": expected_abstentions,
        "predicted_answered": len(citations),
        "recall_at_5": mean(recalls),
        "citation_accuracy": mean(citations),
        "model_revision_accuracy": mean(scopes),
        "unsupported_answer_rate": round(unsupported / len(cases), 4),
        "answers_on_clarify_or_abstain": no_evidence_cases,
        "abstention_recall": round(correct_abstentions / expected_abstentions, 4) if expected_abstentions else None,
        "status_accuracy": round(correct_status / len(cases), 4),
        "p95_ms": round(sorted(timings)[math.ceil(len(timings) * 0.95) - 1], 4),
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Persian retrieval modes on labeled manual questions.")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--semantic", action="store_true", help="Add the configured Ollama hybrid engine.")
    args = parser.parse_args()
    cases = load_cases(args.split)
    engine = Engine()
    result = {
        "split": args.split,
        "runtime": f"Python {platform.python_version()}, {platform.system()} {platform.machine()}",
        "lexical": evaluate(engine, cases, "lexical"),
        "hybrid_rules": evaluate(engine, cases, "hybrid"),
    }
    if args.semantic:
        semantic_engine = Engine(Semantic())
        # Keep model loading and the in-memory index build out of the request timing.
        semantic_engine.semantic.check()
        semantic_engine.ask("نسخه نرم افزار کجاست؟", "CN-120", "1.0")
        result[semantic_engine.retrieval] = evaluate(semantic_engine, cases, "hybrid")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
