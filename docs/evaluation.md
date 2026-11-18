# Persian retrieval benchmarks

## Frozen 40-question challenge

The follow-up includes 20 answerable, seven clarification, and 13 abstention cases against the four fictional manuals. It adds more colloquial paraphrases and requests for unsupported actions or numeric values using real codes from the corpus. The author froze the labels before editing retrieval. The separate `hard-dev.psv` set of 12 questions informed the thresholds and rules. The challenge was authored for this project and has not been reviewed by independent operators.

Challenge SHA-256 with normalized line endings is `b0c64791bbcbc70db9a8ea5400818f589a71f77d848af52f87651172fc746c0e`. The comparison ran on Windows with CPython 3.14.7 and one ONNX Runtime inference thread on an Intel Core i5-9400F. The initial rules were run from release commit `694f913bdab3` on the same challenge. Model loading and corpus indexing are outside warm request latency.

| Challenge metric | Initial rules | Updated rules | Updated rules + multilingual E5-small INT8 |
| --- | ---: | ---: | ---: |
| Correct grounded outcomes | 27/40 (67.5%) | 31/40 (77.5%) | **35/40 (87.5%)** |
| Answerable cases answered with correct evidence | 10/20 | 11/20 | **15/20** |
| Recall@5 | 90% | 95% | **100%** |
| Clarifications with exact missing fields | 7/7 | 7/7 | 7/7 |
| Correct abstentions | 10/13 | **13/13** | **13/13** |
| Unsupported answers | 3/40 | **0/40** | **0/40** |
| Latency p95 | 0.10 ms | 0.10 ms | 10.93 ms |

Grounded outcome accuracy requires the expected state and exact citation IDs or missing fields where applicable. The updated rules normalize Persian plurals and require multiple matching terms, heading support, and separation from other sections before accepting a weaker match. A corpus-specific guard routes recognized requests for unsupported operations and numeric limits to human review before looking up codes. E5 scores the in-scope manual sections with attention-masked mean pooling and cosine similarity. It can supply evidence when rules abstain if the cosine is at least 0.82 and exceeds the next distinct section by at least 0.035. Recall@5 for this mode uses reciprocal rank fusion of lexical and dense ranks.

- **Improved grounding.** Queries `c32`–`c34` mention an existing alarm while asking how to disable it, erase it, or buy a replacement connector. The initial rules returned code descriptions for all three. Both updated modes abstained on all three.
- **Improved retrieval coverage.** The E5 fallback recovered `c03`, a paraphrase of time-based filtering in the alarm history. It also recovered `c06`, `c17`, and `c19` after updated rules abstained.
- **Remaining errors.** E5 abstained on five answerable questions. Examples include `c05`, which uses the colloquial «ریلیز» for software revision, and `c18`, an indirect description of case overheating. The correct citation was ranked in the top five even when acceptance thresholds prevented an answer.

The E5 result used the actual INT8 model, and a live HTTP smoke test covered dense retrieval, exact code lookup, clarification, and abstention. The corpus, queries, and guard rules are specific to this demonstration. Zero unsupported answers applies to this sample. The Ollama model quality and complete Compose stack were outside this follow-up benchmark.

```sh
python evaluate.py --split challenge --e5-model-dir models/multilingual-e5-small
```

## Original 45-question test

The development split contains 15 queries. The test split contains 45 queries, with 26 answerable cases, eight clarification cases, and 11 abstention cases. The two lists contain different questions against the same four authored manuals. The benchmark is synthetic and shares domain vocabulary with the small alias dictionary.

### Initial release results

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

### Initial release error analysis

| Query | Expected | Observed |
| --- | --- | --- |
| t31 — «بیلد را کجا ببینم؟» | Version display for CN-120 2.0 | Lexical matching abstained. Expanding «بیلد» to «نرم افزار» recovered the correct section. |
| t29 — «هشدارای قبلی رو کجا نشون می‌ده؟» | Event history for CN-120 1.0 | The hybrid ranked the correct evidence but its lexical coverage remained below the 0.55 threshold. Colloquial filler still counted against the score. |
| t17 — «کد E۱۲ را توضیح بده» with CN-120 specified | Ask for revision | Matched the Persian-digit code and asked for revision because E12 appears only in 2.0. |
| t45 — «حد دقیق دمای موتور چند درجه است؟» | Human review | Even though the manual describes a motor-temperature warning, neither mode returned a threshold it could not support. |

The table records the initial Python-only release for comparison. The updated rules handled all 45 original test outcomes with correct grounding. The Ollama wire contract was checked against a local HTTP fixture. The pgvector integration test passed against PostgreSQL compiled to WebAssembly through PGlite. [PGlite](https://pglite.dev/docs/about)
