# Persian retrieval benchmark

The development split contains 15 queries. The test split contains 45 queries, with 26 answerable cases, eight clarification cases, and 11 abstention cases. The two lists contain different questions against the same four authored manuals. The benchmark is synthetic and shares domain vocabulary with the small alias dictionary.

## Local test results

One in-process run on Windows with CPython 3.11.16 and an Intel Core i5-9400F. The timings include normalization, retrieval, decision and response construction.

| Metric | Lexical | Hybrid with Persian aliases |
| --- | ---: | ---: |
| Recall@5 | 88.46% | 100% |
| Exact citation accuracy | 100% | 100% |
| Citation model and revision accuracy | 100% | 100% |
| Unsupported answer rate | 0% | 0% |
| Correct abstentions | 11/11 | 11/11 |
| Correct response statuses | 38/45 | 44/45 |
| Answers returned | 19 | 25 |
| Latency p95 | 0.0672 ms | 0.0865 ms |

On development, lexical and hybrid Recall@5 were 77.78% and 100%. The hybrid answered all nine answerable development queries. Run `python evaluate.py --split dev` and `python evaluate.py --split test` to reproduce metrics and list the queries that failed.

- **Recall@5** is the mean fraction of each answerable query's gold evidence retrieved in the first five candidates, before the final answer decision.
- **Citation accuracy** is the fraction of answered queries with exactly their gold evidence IDs. Model and revision accuracy compares only those pairs.
- **Unsupported answer rate** is the fraction of all queries where the system returned an answer to a clarification or abstention case, or cited the wrong evidence for an answerable question.
- **Abstention recall** is the fraction of gold abstention cases the system abstained on. **Status accuracy** checks the three response statuses. The failure list additionally checks the requested metadata for clarification.

## Error analysis

| Query | Expected | Observed |
| --- | --- | --- |
| t31 — «بیلد را کجا ببینم؟» | Version display for CN-120 2.0 | Lexical matching abstained. Expanding «بیلد» to «نرم افزار» recovered the correct section. |
| t29 — «هشدارای قبلی رو کجا نشون می‌ده؟» | Event history for CN-120 1.0 | The hybrid ranked the correct evidence but its lexical coverage remained below the 0.55 threshold. Colloquial filler still counted against the score. |
| t17 — «کد E۱۲ را توضیح بده» with CN-120 specified | Ask for revision | Matched the Persian-digit code and asked for revision because E12 appears only in 2.0. |
| t45 — «حد دقیق دمای موتور چند درجه است؟» | Human review | Even though the manual describes a motor-temperature warning, neither mode returned a threshold it could not support. |

The table measures the Python-only configurations. The Ollama wire contract was checked against a local HTTP fixture. The pgvector integration test passed against PostgreSQL compiled to WebAssembly through PGlite. Real embedding and generation quality and the complete Compose stack were unmeasured on this host. [PGlite](https://pglite.dev/docs/about)
