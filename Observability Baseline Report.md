# Phase 4 — Initial Observability & Performance Baseline Report

## 1. Measurement Configuration
- Knowledge Base: 8 documents (122 chunks stored in `data/chroma_db`, collection `rag_knowledge_base`)
- Embeddings & Retrieval: `nomic-embed-text:latest`, `k=4`
- LLM Generation: Ollama `llama3.2:3b`, `temperature=0.0`, blocking mode (`stream=False`)
- Evaluation Dataset: 12 benchmark queries (`EVAL-001` through `EVAL-012`), 12/12 succeeded (`0` failed)
- Persisted Artifacts: `data/telemetry/baseline_metrics.jsonl` and `data/telemetry/baseline_report.json`

## 2. Phase 2 vs. Phase 4 Baseline Comparison
- **Retrieval Latency:**
  - Phase 2 Baseline: ~90 ms mean (~1.09 s total across 12 queries).
  - Phase 4 Observed: 166.65 ms mean (102.00 ms median, 2.00 s total across 12 queries).
  - Analysis: `EVAL-001` experienced an 854.5 ms cold-start spike while loading `nomic-embed-text:latest` into memory. Excluding that single cold start, the remaining 11 queries averaged 104.12 ms (median 102.00 ms), matching the Phase 2 baseline closely.
- **End-to-End Latency:**
  - Phase 2 Baseline: ~50.07 s mean (~600.87 s total across 12 queries).
  - Phase 4 Observed: 64.06 s mean (67.24 s median, 768.70 s total across 12 queries; min 29.36 s on `EVAL-011`, max 79.34 s on `EVAL-002`).
  - Analysis: Both baselines confirm multi-dozen-second local CPU/RAM generation times on `llama3.2:3b`, with normal variance attributable to local system load and thermal/memory conditions during execution.
- **New Metrics Established in Phase 4:**
  - Context Construction: 0.03 ms mean (min 0.02 ms, max 0.05 ms); mean size 4,319.50 chars.
  - Prompt Construction: 0.01 ms mean across all queries.
  - Token Usage: Mean 1,111.33 total tokens per request (min 1,025 on `EVAL-012`, max 1,207 on `EVAL-008`), with prompt tokens averaging ~969 and output tokens ranging from 55 (`EVAL-012`) to 242 (`EVAL-008`).
  - TTFT: Explicitly recorded as `None` (`n=0`) since blocking mode returns the full completion at once.
  - Statistical Guardrails: `P90` calculated at `n=12` (78,683.30 ms total); `P95` reported as `N/A` because `n < 20`.

## 3. Bottleneck Analysis & Evidence-Based Conclusions
- **Primary Bottleneck — LLM Generation (99.74% of Total Latency):**
  - Generation consumes 63,891.86 ms out of 64,058.59 ms mean total latency.
  - Every query carries ~900–1,011 input tokens (mostly from 4 full context chunks averaging 4,319.50 chars) plus 55–242 generated output tokens on local hardware.
  - Shorter answers (`EVAL-011`: 63 output tokens in 29.36 s; `EVAL-012`: 55 output tokens in 40.29 s) finish noticeably faster than longer answers (`EVAL-002`: 187 output tokens in 79.34 s; `EVAL-008`: 242 output tokens in 74.88 s).
- **Non-Bottlenecks (< 0.3% Combined):**
  - **Retrieval (0.26% of Total Latency):** 166.65 ms mean (102.00 ms median).
  - **Context & Prompt Construction (< 0.001% of Total Latency):** 0.04 ms combined mean.
- **Guidance for Subsequent Phases (Phase 5 & Phase 6/7):**
  - Do not spend optimization effort on context formatting, prompt string construction, or ChromaDB search speed.
  - Focus future optimization hypotheses strictly on generation-side and context-token efficiency (e.g., streaming for TTFT responsiveness, response caching for repeated queries, context size/chunking reduction to shrink ~1,000-token prefill cost, and concise output control).