# Persian Maintenance RAG

Persian question answering for two fictional conveyor controllers, each with two manual revisions. The API shows the source model, revision, section, and manual excerpt. It asks for missing metadata when the manual interpretation changes and flags unsupported queries for human review.

سامانهٔ جست‌وجو در راهنمای فارسی دو کنترلر فرضی تسمه‌نقاله. برای نمونه، معنی `E17` را در هر دو مدل بپرسید و سپس همان پرسش را بدون مدل و نسخه بفرستید.

## Run the demo

~~~sh
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
~~~

Open the [Persian demo](http://127.0.0.1:8000/) or the [interactive API docs](http://127.0.0.1:8000/docs). The demo has three E17 example buttons. The first two send the same question about `CN-120` and `CN-240` at revision `2.0`. The third asks it without metadata.

| Request | Result |
| --- | --- |
| E17 on CN-120, 2.0 | The measured belt speed differs from the reference. |
| E17 on CN-240, 2.0 | The controller stopped receiving diagnostic messages from the output sensor. |
| E17 with no model or revision | Requests both fields. |
| E999 | Marks the question for human review. |

The quickstart uses lexical matching and a small Persian alias dictionary. It works with Python alone. The API documents machine-readable statuses and evidence. [API contract](docs/api.md)

## Run the local model and pgvector

~~~sh
docker compose up --build
~~~

Compose starts PostgreSQL 16 with pgvector, pulls `embeddinggemma` and `qwen2.5:1.5b` inside Ollama, indexes the four manuals, and serves the same demo. The first run needs network access for the image and model downloads. PostgreSQL and Ollama stay inside the Compose network. The API is bound to localhost. [EmbeddingGemma](https://ollama.com/library/embeddinggemma) · [Qwen2.5 1.5B](https://ollama.com/library/qwen2.5:1.5b) · [pgvector](https://github.com/pgvector/pgvector)

To use an installed Ollama directly, start Ollama separately and pull both models.

~~~sh
ollama pull embeddinggemma
ollama pull qwen2.5:1.5b
~~~

Set `OLLAMA_URL=http://127.0.0.1:11434` in the API process environment. For PowerShell, run `$env:OLLAMA_URL = 'http://127.0.0.1:11434'`. Start Uvicorn as above. This mode keeps the 32 manual vectors in memory. Set `DATABASE_URL` and install `requirements-semantic.txt` to use a separate PostgreSQL instance. Run `python semantic.py` once to index the manuals before starting the API.

## How requests are answered

1. Normalize Persian and Arabic digits, Arabic forms of ی and ک, diacritics, and half-spaces. Resolve model and revision from request fields or the question.
2. Scope candidates by metadata. When the question names one code, match the exact error row. Compare its excerpt across admissible manuals and ask for the metadata that changes the result.
3. For prose questions, combine token overlap with alias expansion in the quickstart or cosine similarity over multilingual embeddings in Ollama mode. Ollama can select a shortlisted evidence ID or return an abstention. The API validates the ID and reproduces the source excerpt verbatim.

The corpus is four authored Markdown manuals with 32 sections and error rows. The demo uses local fictional data. A human-review flag is returned in the response for a caller to handle.

## Evaluation

~~~sh
python evaluate.py --split dev
python evaluate.py --split test
docker compose exec api python evaluate.py --split test --semantic
~~~

The benchmark compares lexical matching and the alias-based hybrid on 15 development queries and 45 test queries. Use `--semantic` in either model-backed configuration to add the embedding and model result on the same questions. [Metrics and failure examples](docs/evaluation.md) · [Labeled splits](evaluation/)

## Tests

~~~sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
~~~

The tests cover the public API, allowed evidence IDs, the Ollama HTTP contract, and the benchmark labels. The pgvector integration test runs in GitHub Actions with a real PostgreSQL/pgvector service and a deterministic local Ollama HTTP fixture.
