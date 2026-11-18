# Persian Maintenance RAG API

`POST /ask` accepts `{"question":"کد E17 چیست؟","model":"CN-240","revision":"2.0"}`. Only `question` is required. It is a nonblank string of at most 500 characters. `model` accepts `CN-120` or `CN-240` and `revision` accepts `1.0` or `2.0`. The same model and revision can be written inside the question. Conflicting values prompt clarification. A query with exactly one error code requests its documented definition. Invalid fields or types return 422.

All successful queries return HTTP 200 with `status`, `answer`, `citations`, `missing`, and `human_review`.

| Status | `answer` | `citations` | `missing` | `human_review` |
| --- | --- | --- | --- | --- |
| `answered` | Persian manual excerpt | One or more evidence objects | `[]` | `false` |
| `clarify` | Persian follow-up question | `[]` | Model and/or revision field names | `false` |
| `abstain` | Persian handoff explanation | `[]` | `[]` | `true` |

Evidence objects contain `id`, `title`, `model`, `revision`, `section`, `code` (null for prose sections), and `excerpt`. Evidence IDs identify a single manual section or error-code row. `GET /health` returns `status` and `retrieval`. In dependency-backed mode, a failed check or query returns 503. `GET /` opens the Persian demo. `GET /docs` exposes the generated OpenAPI contract and an interactive request form.

`GET /health` identifies `hybrid-rules`, `hybrid-e5`, `hybrid-ollama` or `hybrid-pgvector`. In `hybrid-e5` the local ONNX model loads at startup. All modes return the same grounded response shape. A recognized request for a procedure or numeric limit outside the fictional manuals is passed to human review even if it mentions a valid code.
