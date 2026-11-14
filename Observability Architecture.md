# Observability Architecture & Measurement Design

## 1. Architectural Layers & Boundaries
To ensure our core RAG components remain pure, modular, and testable, we enforce strict separation across five distinct layers:

1. **RAG Application Layer:** Core business logic (RAGPipeline, Retriever, VectorStore, ContextBuilder, PromptBuilder, AnswerGenerator). Responsible only for retrieval, context assembly, prompt formatting, and generation; completely unaware of telemetry persistence or logging backends.
2. **Instrumentation Layer:** Captures execution events, monotonic timings, and component-level metrics via lightweight wrappers, decorators, or contextual execution tracking.
3. **Observability / Telemetry Layer:** Manages request correlation (request_id), coordinates measurement collection, and structures telemetry records according to the agreed schema.
4. **Logging Layer:** Handles structured operational logs, execution lifecycle milestones, and exception tracing, isolated from performance metrics aggregation.
5. **Persistence & Reporting Layer:** Manages lightweight storage (local JSON / SQLite records), retrieval of historical traces, and statistical aggregation (mean, median, P90, P95).

---

## 2. Request Identification & Propagation Flow
Every execution triggered through RAGPipeline.query(...) must be uniquely identifiable and traceable.

- **Generation:** A unique UUID (UUIDv4) is generated at the entry point of RAGPipeline.query() unless supplied externally by an upstream caller (e.g., API layer).
- **Propagation Mechanism:** Uses Python’s standard library contextvars.ContextVar.
  - Avoids polluting component method signatures across the business domain (retrieve, build_context, generate) with non-domain parameters.
  - Ensures thread-safe and async-safe propagation across the call hierarchy.
  - Allows log formatters and instrumentation hooks to automatically associate the active request_id with all emitted events.

---

## 3. Request-Level Measurement Schema
Each RAG invocation produces a consolidated telemetry record structured as follows:

```
{
"request_id": "uuid-v4",
"timestamp": "ISO-8601 UTC",
"configuration": {
"app_version": "git-commit-hash",
"embedding_model": "nomic-embed-text:latest",
"llm_model": "llama3.2:3b",
"k": 4
},
"retrieval": {
"retrieval_latency_ms": 0.0,
"k_configured": 4,
"documents_retrieved": 0,
"status": "success"
},
"context": {
"context_construction_latency_ms": 0.0,
"chunks_used": 0,
"context_size_chars": 0,
"context_size_tokens_estimated": 0
},
"prompt": {
"prompt_construction_latency_ms": 0.0
},
"llm": {
"model": "llama3.2:3b",
"generation_latency_ms": 0.0,
"ttft_ms": null,
"prompt_tokens": 0,
"generation_tokens": 0,
"total_tokens": 0,
"status": "success"
},
"error": {
"occurred": false,
"stage": null,
"message": null
},
"total_latency_ms": 0.0
}
```
---

## 4. Core Metric Catalogue

| Component | Metric | Unit / Type | Description |
| :--- | :--- | :--- | :--- |
| **Pipeline** | request_id | String (UUIDv4) | Unique correlation identifier for tracing. |
| **Pipeline** | total_latency_ms | Float (ms) | Monotonic wall-clock time for the full query workflow. |
| **Retrieval** | retrieval_latency_ms | Float (ms) | Time spent querying ChromaDB and retrieving chunks. |
| **Retrieval** | documents_retrieved | Integer | Total number of chunks returned by the similarity search. |
| **Context** | context_construction_latency_ms | Float (ms) | Time spent formatting and ordering context chunks. |
| **Context** | chunks_used | Integer | Number of chunks incorporated into the final context. |
| **Context** | context_size_chars | Integer | Total character length of the assembled context string. |
| **Prompt** | prompt_construction_latency_ms | Float (ms) | Time spent formatting system and user templates. |
| **LLM** | generation_latency_ms | Float (ms) | Total generation time elapsed during model inference. |
| **LLM** | ttft_ms | Float (ms) or Null | Time to First Token (measured during streaming when active). |
| **LLM** | prompt_tokens | Integer or Null | Prompt token count reported by Ollama. |
| **LLM** | generation_tokens | Integer or Null | Completion token count reported by Ollama. |
| **LLM** | total_tokens | Integer or Null | Aggregate token count for prompt and completion. |

---

## 5. Data Privacy & Operational Boundaries
To ensure user privacy, security compliance, and manageable log volumes:

- **Strictly Excluded from Telemetry & Logs:**
  - Raw user prompt text and query questions.
  - Raw document text and chunk bodies.
  - Generated completion body text.
- **Permitted in Telemetry & Logs:**
  - Correlation identifiers (request_id).
  - Monotonic timings and durations (ms).
  - Quantitative metrics (counts, character lengths, token tallies).
  - Component configuration metadata (model names, k).
  - Error types and sanitized exception descriptions.

---

## 6. Structured Logging Standards
- Configured using standard Python logging.
- Operational events are logged with structured contextual information (request_id, component name, duration).
- Standard log levels:
  - INFO: Lifecycle milestones (e.g., query started, retrieval completed, generation completed, request finished).
  - DEBUG: Internal counts, chunk metadata, intermediate timings.
  - WARNING / ERROR: Component failures, empty retrieval results, generation exceptions.

---

## 7. Implementation Roadmap (Phase 4 Progression)
- **Milestone 1:** Observability Architecture & Measurement Design (This document).
- **Milestone 2:** Request Identification and Structured Logging (request_id context and structured loggers).
- **Milestone 3:** Component-Level Timing and Latency Instrumentation (monotonic timers and context metrics).
- **Milestone 4:** LLM Observability and Token Measurement (Ollama generation metrics and token counts).
- **Milestone 5:** Metrics Collection, Aggregation, and Reporting (local persistence and statistical summaries).
- **Milestone 6:** Observability Validation and Performance Baseline (telemetry verification against evaluation dataset).