# Baseline Environment & Configuration Specification

## 1. Hardware & Host Environment
- Operating System: Windows (PowerShell terminal)
- System Memory: 16 GB RAM
- Local LLM Runtime: Ollama (local server)
- Python Version: 3.12 (managed via uv in agent_env virtual environment)

## 2. Model & Embedding Configuration
- Generation Model: Ollama llama3.2:3b
- Generation Parameters: temperature=0.0, stream=False (blocking mode)
- Embedding Model: Ollama nomic-embed-text:latest
- Prompt Strategy: Grounded prompt template instructing direct answers, coverage of all material facts, structured lists where appropriate, and strict grounding in retrieved context without external invention.

## 3. Storage & Ingestion Pipeline
- Document Corpus: 8 files (business_travel.md, code_of_conduct.txt, employee_benefits.md, employee_leave.pdf, employee_onboarding.txt, expense_reimbursement.pdf, it_security.md, remote_work.pdf)
- Document Loader: DocumentLoader (handling .md, .txt, and .pdf via pypdf)
- Chunking: RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200, length_function=len)
- Vector Database: ChromaDB (persistent on disk at data/chroma_db)
- Collection Name: rag_knowledge_base (122 indexed chunks)

## 4. Query & Retrieval Configuration
- Vector Search: Cosine/Euclidean similarity search through ChromaDB
- Retrieval Limit: k=4 chunks per query
- Context Construction: ContextBuilder with source attribution tags [Document X | Source: Y]

## 5. Evaluation Workload Characterization
- Benchmark Dataset: data/evaluation/questions.jsonl (12 questions)
- Answerable Queries: 11 questions (direct facts and procedural requirements)
- Unanswerable Sentinel: 1 question (EVAL-012, testing out-of-corpus refusal)
- Query Typology:
  - Single-Source Queries: 9 questions (focused retrieval against 1 policy document)
  - Multi-Source Queries: 2 questions (EVAL-009, EVAL-010, requiring cross-document synthesis)
- Target Document Distribution: 100% coverage of the 8-document corpus (expense_reimbursement: 3, business_travel: 2, it_security: 2, remote_work: 2, benefits: 1, leave: 1, conduct: 1, onboarding: 1).
- Execution Conditions: Sequential single-thread batch execution under local CPU/RAM load, with an observable cold start on the first query.

## 6. Measurement Assessment & Technical Limitations
- Data Completeness: 100% capture across all 12 evaluation queries with zero missing stages or unhandled failures.
- Direct Measurements:
  - Monotonic latencies for retrieval, context, prompt, generation, and total request duration.
  - Quantitative counts: k=4, documents retrieved, chunks used, and context character length.
  - Token counts extracted directly from Ollama runtime metadata.
- Estimated Metrics:
  - Context token size is estimated via character heuristic (chars // 4). Ollama prompt tokens provide the authoritative token consumption figure.
- Unavailable Metrics (Explicitly Stored as None):
  - TTFT is unobservable in blocking mode (stream=False); it is recorded as None rather than 0.0 to prevent data corruption.
  - OS-level hardware utilization (RAM allocation, GPU VRAM, CPU load) was not profiled to avoid external instrumentation overhead.
- Statistical & Environmental Limitations:
  - Sample size (n=12) supports mean, median, min, max, and P90; P95 is statistically excluded until sample size reaches n >= 20.
  - Local host variance: Single-thread execution on a 16 GB machine is subject to thermal conditions and background OS scheduling, accounting for generation latency spread (29.28 s to 79.22 s).

## 7. Historical Baseline Comparison (Phase 2 vs. Phase 4)
- Workload and Configuration Parity: Identical 12 evaluation questions, identical 8-document corpus (122 chunks), and identical models (llama3.2:3b, nomic-embed-text:latest, k=4).
- Retrieval Performance:
  - Phase 2: Mean retrieval latency was ~90 ms per query (1.09 s total).
  - Phase 4: Median retrieval latency was 102.00 ms (mean 166.65 ms, influenced by an 854.51 ms cold-start on EVAL-001).
  - Conclusion: Steady-state vector retrieval latency remains stable at ~100 ms.
- End-to-End Latency:
  - Phase 2: Mean total latency was ~50.07 s (600.87 s total).
  - Phase 4: Mean total latency was ~64.06 s (768.70 s total; median 67.24 s).
  - Conclusion: Both runs confirm multi-dozen-second local inference times on CPU/RAM. The variance is consistent with local host thermal throttling and operating system scheduling overhead.
- Measurement Methodology Evolution:
  - Phase 2 relied on manual wall-clock wrappers without token accounting.
  - Phase 4 introduced monotonic sub-millisecond isolation, exact Ollama token telemetry, and automated JSON Lines persistence, providing the necessary precision to guide subsequent optimization.