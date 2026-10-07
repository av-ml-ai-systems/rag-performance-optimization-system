# RAG Performance Optimization System

A modular Retrieval-Augmented Generation (RAG) system focused on measuring, analyzing, and optimizing retrieval, generation, latency, token usage, and resource efficiency.

## Project Objective

The project explores how to systematically improve the performance and resource efficiency of a RAG system while preserving useful behavior and maintaining clean, maintainable software architecture.

The core engineering loop is:

**Build → Measure → Understand → Identify Bottleneck → Form Hypothesis → Optimize → Measure Again → Validate → Keep or Reject**

## Scope

The system will progressively cover:

* Document ingestion and processing
* Text chunking
* Embedding generation
* Vector storage and retrieval
* LLM-based generation
* Performance measurement and observability
* Retrieval and generation optimization
* API exposure
* A lightweight performance dashboard
* Regression and optimization validation

The baseline RAG implementation will intentionally remain simple. Optimization will be introduced only after establishing measurable baseline behavior.

## Technology Stack

* Python
* LangChain
* Hugging Face / Sentence Transformers
* ChromaDB
* Ollama
* FastAPI
* React
* pytest
* Ruff
* MyPy
* Pyright
* pre-commit
* Git / GitHub

## Development Environment

The project uses a Conda environment managed through `uv`.

The project dependency workflow is:

**`pyproject.toml → uv sync → uv.lock → Conda environment`**

Development dependencies are defined in `pyproject.toml`.

## Development Checks

The main quality tools are:

```text
ruff check .
ruff format --check .
mypy .
pyright
pytest
pre-commit run --all-files
```

## Project Structure

The project follows a modular architecture that will evolve incrementally as the RAG system is implemented.

```text
src/
└── rag_performance/

tests/
```

RAG-specific boundaries will be introduced during implementation rather than created prematurely.

## Project Blueprint

The complete learning strategy, architecture, roadmap, testing strategy, Git workflow, definition of done, and scope-control rules are documented in:

`Blueprint.md`

## Project Status

**Phase 1 — Project Foundation**

The repository, development environment, dependency management, code-quality tooling, type checking, testing foundation, and pre-commit workflow are being established before implementing the baseline RAG system.
