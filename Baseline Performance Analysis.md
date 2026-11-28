# Baseline Performance Analysis

## 1. Consolidated Baseline Measurements (Phase 4 Reference)
- Workload: 12 evaluation queries (EVAL-001 through EVAL-012)
- Execution Status: 12/12 successful (0 failures)
- Total Request Latency:
  - Mean: 64,058.59 ms (~64.06 s)
  - Median: 67,242.75 ms (~67.24 s)
  - Minimum: 29,356.87 ms (~29.36 s)
  - Maximum: 79,335.77 ms (~79.34 s)
  - P90: 78,683.30 ms (~78.68 s)
  - P95: N/A (sample size n=12 < 20)
- Component Latency Breakdown:
  - LLM Generation: mean 63,891.86 ms (99.74% of total time)
  - Document Retrieval: mean 166.65 ms (median 102.00 ms, min 75.65 ms, max 854.51 ms cold-start)
  - Context Construction: mean 0.03 ms
  - Prompt Construction: mean 0.01 ms
  - Time to First Token (TTFT): N/A (blocking mode)
- Token Consumption & Context Footprint:
  - Context Size: mean 4,319.50 characters (min 4,053, max 4,485)
  - Prompt Tokens: mean 965.33 tokens (min 904, max 1,011)
  - Generation Tokens: mean 146.00 tokens (min 55, max 242)
  - Total Tokens: mean 1,111.33 tokens (min 1,025, max 1,207)

## 2. Baseline Environment & Configuration Specification
- Hardware Host: Local Windows workstation, 16 GB RAM
- Local LLM Runtime: Ollama local server
- Python Environment: Python 3.12 managed via uv (agent_env)
- Generation Model: Ollama llama3.2:3b
  - Parameters: temperature=0.0, stream=False (blocking mode)
- Embedding Model: Ollama nomic-embed-text:latest
- Ingestion & Storage:
  - Corpus: 8 corporate policy documents (.md, .txt, .pdf)
  - Chunking: RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200, length_function=len)
  - Vector Database: ChromaDB persistent at data/chroma_db
  - Collection: rag_knowledge_base (122 indexed chunks)
- Retrieval Configuration:
  - Similarity Search: Cosine/Euclidean similarity against vector embeddings
  - Top-k Limit: k=4 chunks per query
- Prompt Strategy: Strict context-grounded template instructing direct factual answers, complete coverage of requirements, and explicit admission of insufficient information.