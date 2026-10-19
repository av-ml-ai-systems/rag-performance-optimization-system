# RAG Performance Optimization System

## Project Roadmap & Learning Blueprint

**Version:** Revised Blueprint  
**Core focus:** RAG Engineering, Object-Oriented Programming, Software Engineering, Performance Optimization, Testing, and Evidence-Based Engineering.

---

# 1. Project Definition

## 1.1 Project Identity

- **Project name:** RAG Performance Optimization System
- **Repository:** `rag-performance-optimization-system`
- **Project type:** Practical AI/ML engineering project focused on building, measuring, analyzing, and optimizing a modular Retrieval-Augmented Generation (RAG) system.

## 1.2 Central Engineering Question

> Can we build a clean, modular, object-oriented RAG system and systematically measure and optimize its performance and resource usage without unnecessarily sacrificing answer quality or useful behavior?

## 1.3 Core Engineering Loop

The project begins with a deliberately simple **Naive RAG baseline**. Optimization follows only after the system's behavior has been measured and understood.

**Build → Measure → Understand → Identify Bottleneck → Form Hypothesis → Optimize → Measure Again → Validate Quality → Keep or Reject**

An optimization is successful only when its benefits justify its trade-offs in quality, resource consumption, maintainability, and complexity.

## 1.4 Project Dimensions

The project combines:

- RAG engineering and local LLM integration.
- Python and object-oriented software engineering.
- Clean Code and SOLID principles.
- Testing, static analysis, and maintainable architecture.
- Performance measurement and observability.
- Controlled optimization experiments.
- FastAPI backend and lightweight React interface.
- Git/GitHub workflow and professional documentation.

---

# 2. Project Objectives

## 2.1 RAG Engineering Objective

Build a functional, modular RAG application using:

- LangChain.
- Hugging Face / Sentence Transformers.
- ChromaDB.
- Ollama.
- A practical local LLaMA-family model.

The baseline must support document ingestion, chunking, embedding generation, vector storage, retrieval, context and prompt construction, LLM generation, and returning answers with useful retrieval information.

The objective is to understand and establish a controlled RAG system, not to implement the most sophisticated architecture possible.

## 2.2 Performance Optimization Objective

Investigate meaningful performance and resource dimensions, selected according to baseline evidence.

**LLM and generation candidates:**

- Token usage and context size.
- Caching and streaming.
- Time to First Token (TTFT) and generation latency.
- Resource efficiency.
- Model selection or routing, when justified.

**RAG and retrieval candidates:**

- Chunking efficiency.
- Retrieval parameters and vector-search behavior.
- Embedding efficiency and resource usage.
- Retrieval latency and end-to-end RAG latency.

Not every candidate must be implemented. Experiments are selected according to their expected value, measured bottlenecks, and implementation cost.

## 2.3 Software Engineering Objective

Professional software engineering is a **first-class project objective**, not merely a supporting activity.

The implementation must apply the following principles pragmatically:

- Object-Oriented Programming (OOP) from the beginning of Phase 2.
- Clean Code and SOLID principles.
- Separation of concerns, high cohesion, and low coupling.
- Encapsulation, abstraction, and polymorphism where useful.
- Composition and dependency injection where justified.
- Dependency inversion and appropriate interfaces or protocols.
- Clear class responsibilities and module boundaries.
- Type hints, meaningful naming, focused functions, and cohesive modules.
- Explicit configuration, appropriate exception handling, logging, and testability.
- DRY and KISS, without premature abstraction or unnecessary complexity.

The project will also use Git/GitHub, pytest, Ruff, MyPy, Pyright, pre-commit, automated quality checks, meaningful commits, and clear documentation.

**Engineering principle:** The goal is not to maximize the number of classes or abstractions. The goal is clean, understandable, maintainable, testable software designed appropriately for the actual problem.

## 2.4 Learning Objective

By the end of the project, the user should be able to:

- Explain the complete RAG architecture and each major technology.
- Understand and explain the important Python classes, functions, modules, and dependencies.
- Design cohesive classes with clear responsibilities.
- Explain encapsulation, abstraction, composition, dependency injection, and relevant SOLID principles.
- Recognize when an abstraction helps and when it introduces unnecessary complexity.
- Apply Clean Code, OOP, and software testing in a real project.
- Measure performance, identify bottlenecks, and design controlled experiments.
- Interpret results and performance/quality trade-offs.
- Defend architectural and optimization decisions in a technical interview.

The project must produce both **a working system** and **a strong understanding of how and why it was built**.

---

# 3. Scope and Boundaries

## 3.1 Core Scope

The project includes:

- A working Naive RAG system.
- Document ingestion, chunking, embeddings, ChromaDB, retrieval, and Ollama generation.
- Meaningful use of LangChain and Hugging Face / Sentence Transformers.
- Object-oriented Python architecture and professional software engineering.
- Automated testing, static analysis, and quality checks.
- Baseline measurement, observability, and selected optimization experiments.
- FastAPI backend and lightweight React interface.
- Regression validation, Git/GitHub workflow, and professional documentation.

## 3.2 Software Engineering Scope

OOP, Clean Code, SOLID, modularity, dependency management, type safety, configuration, error handling, logging, testing, and maintainable Git history are explicit project requirements.

These practices will evolve progressively as requirements and real implementation needs become clear.

The baseline will be designed with appropriate class responsibilities and component boundaries from the beginning. Phase 3 will strengthen and validate the architecture rather than introduce OOP for the first time.

## 3.3 Conditional and Experimental Scope

The following are optional candidates, not initial requirements:

- Response caching and streaming.
- Model or embedding-model comparison.
- Model routing.
- Context compression.
- Retrieval parameter and chunking optimization.
- Additional retrieval experiments and performance instrumentation.

Their implementation depends on measured evidence, available time, learning value, and whether the benefits justify the complexity.

## 3.4 Out of Scope

The core project excludes:

- Agentic RAG, GraphRAG, multi-agent systems, and LangGraph.
- Hybrid search, complex query rewriting, and multi-query retrieval.
- Reranking unless a strong project-specific justification emerges.
- Fine-tuning and multimodal RAG.
- Cloud deployment, Kubernetes, distributed systems, and enterprise-scale architecture.
- Full evaluation platforms, RAGAS, and LLM-as-a-judge infrastructure.
- Comprehensive guardrail systems.
- LangSmith as a required dependency.

The project focuses on **RAG performance engineering and optimization**, not implementing every available RAG technique.

---

# 4. Learning Strategy

## 4.1 Learning Ownership

**The user writes, understands, and owns the Python code.**

ChatGPT provides explanations, architectural reasoning, focused examples, implementation guidance, debugging assistance, code review, testing guidance, and engineering feedback.

Large blocks of opaque generated code should be avoided.

## 4.2 Implementation Learning Loop

Every meaningful implementation step follows:

**Understand → Design → Small Implementation Step → User Implements → Run → Test → Review → Fix and Understand → Quality Checks → Commit**

Progression depends on validating the current step.

## 4.3 Python and OOP Learning

The project will progressively teach and apply:

- Python functions, classes, modules, and packages.
- Encapsulation, abstraction, inheritance when appropriate, and polymorphism when useful.
- Composition and class relationships.
- Interfaces, protocols, and dependency injection.
- Type hints and static type checking.
- Exceptions, configuration, and file organization.
- Unit and integration testing.
- Refactoring, maintainability, and modular design.

Every important class must have a clear responsibility. Dependencies should be explicit and replaceable when there is a concrete benefit, especially for testing and experimentation.

**OOP does not mean turning every function or data structure into a class.** The design should use classes where they improve cohesion, encapsulation, replaceability, or testability.

Unfamiliar Python constructs and design decisions must be explained before becoming part of the implementation.

## 4.4 Software Engineering Learning

For every important design decision, ask:

- Why is it necessary?
- What problem does it solve?
- What complexity does it introduce?
- When should the principle be applied?
- When would a simpler approach be better?
- How does it affect maintainability and testability?

SOLID and Clean Code are engineering tools, not mechanical checklists.

## 4.5 Scope Control and Validation

Every proposed addition should directly support a project objective, provide significant learning value, validate an important hypothesis, or solve a concrete problem.

Each phase requires explicit technical and learning validation. The fact that code runs is not sufficient.

Validation may include functional tests, unit and integration tests, quality checks, retrieval checks, performance measurements, regression checks, manual review, architectural review, and demonstrated understanding.

---

# 5. Technical Architecture

## 5.1 Architectural Objective

Establish clear boundaries between:

- User interface.
- API layer.
- Application and use-case logic.
- Domain concepts and contracts.
- RAG components.
- Infrastructure implementations.
- Observability and optimization mechanisms.

The architecture must remain simple, understandable, testable, and extensible without unnecessary enterprise complexity.

## 5.2 High-Level Architecture

The initial architectural direction is:

```text
React UI
   │ HTTP
   ▼
FastAPI — API Layer
   ▼
Application Layer — Use Cases
   ▼
Domain Concepts / Contracts
   ▲
   │ Implementations
Infrastructure / RAG Components
   ├── LangChain
   ├── ChromaDB
   ├── Hugging Face / Sentence Transformers
   ├── Ollama
   └── Optional Cache
```

This is a conceptual model. The precise package and module structure will be confirmed during implementation.

## 5.3 Baseline RAG Data Flow

```text
Documents
    ↓
Document Loading
    ↓
Chunking
    ↓
Embedding Generation
    ↓
ChromaDB
    ↓
Retriever
    ↓
Retrieved Context
    ↓
Prompt Construction
    ↓
Ollama / Local LLM
    ↓
Answer + Retrieval Information
```

## 5.4 Application Request Flow

```text
User Question
    ↓
FastAPI Endpoint
    ↓
Application Use Case
    ↓
RAG Pipeline
    ↓
Retrieval and Context Construction
    ↓
LLM Generation
    ↓
Performance Measurement
    ↓
RAG Result
    ↓
API Response
    ↓
React UI
```

API route handlers must remain thin. They validate and coordinate requests, then delegate application behavior to appropriate use cases.

## 5.5 Object-Oriented Design and Component Responsibilities

OOP must be applied from the beginning of Phase 2.

The initial design process must identify class responsibilities, relationships, module boundaries, dependency direction, and testability before implementation.

Potential component responsibilities include:

- **Document loading:** read and normalize source documents.
- **Chunking:** divide documents into appropriate text segments.
- **Embeddings:** generate vector representations of text.
- **Vector storage:** persist embeddings and metadata.
- **Retrieval:** find relevant chunks for a query.
- **Context construction:** assemble retrieved information.
- **Prompt construction:** prepare the input supplied to the LLM.
- **Generation:** invoke the configured local model.
- **RAG pipeline:** coordinate the end-to-end workflow.

These are responsibilities, not a mandatory one-class-per-operation rule. The final class design must be documented and approved before implementation.

Use encapsulation, composition, and dependency injection when they provide clear value. Avoid unnecessary inheritance hierarchies, factories, interfaces, or abstractions created solely to follow a pattern.

## 5.6 Architectural Layers

**API layer**

- HTTP endpoints, request validation, response schemas, and HTTP-level errors.
- No direct implementation of RAG workflows, prompt construction, or application rules.

**Application layer**

- Use cases, workflow coordination, and execution control.
- Potential use cases include `AskQuestion`, `IngestDocuments`, `RunExperiment`, and `RetrieveExperimentResults`, subject to actual requirements.

**Domain layer**

- Framework-independent concepts and contracts where justified.
- Possible concepts include `Question`, `RetrievedDocument`, `RetrievedContext`, `Answer`, `RAGResult`, and `ExperimentResult`.
- These are candidates, not mandatory classes.

**Infrastructure layer**

- Integrations with LangChain, ChromaDB, Hugging Face, Ollama, the filesystem, persistence, and optional caching.
- External implementation details should not unnecessarily leak into the application layer.

## 5.7 Dependency Direction and Inversion

Prefer dependencies on stable contracts where doing so improves the design.

```text
API
 ↓
Application
 ↓
Domain / Contracts
 ↑
Infrastructure Implementations
```

Infrastructure implementations may satisfy contracts needed by the application. The application should not become tightly coupled to a specific vector database, LLM runtime, or framework implementation without a good reason.

Dependency inversion and interfaces must be introduced selectively. Not every class needs an interface.

## 5.8 Observability and Optimization Boundaries

Keep measurement and optimization mechanisms separate from core RAG responsibilities where practical.

Potential measurement dimensions include latency, token usage, context size, errors, and resource consumption. Potential optimization mechanisms include caching, retrieval configuration, context reduction, and streaming.

The architecture should support controlled experiments without requiring a rewrite of the entire system.

## 5.9 Architecture Evolution

Follow an evolutionary design approach:

**Simple requirement → Simple implementation → Real problem discovered → Justified abstraction or refactor → Test → Continue**

Do not build a large abstraction hierarchy to anticipate every hypothetical future requirement.

## 5.10 Baseline-First Rule

Do not introduce caching, routing, advanced compression, complex retrieval, or elaborate abstractions merely because they are available. Establish a working and measurable baseline first.

---

# 6. Technology Stack

## 6.1 Python

The primary language for RAG components, application logic, infrastructure integrations, optimization, observability, FastAPI, testing, and automation.

The implementation must strengthen Python and software engineering skills through understandable, maintainable code.

## 6.2 LangChain

Use LangChain meaningfully for appropriate RAG utilities, such as document loading, text splitting, embeddings integration, retrievers, prompt templates, runnable composition, model integration, and Ollama integration.

For important abstractions, understand what they provide, what complexity they hide, why they are useful, and when direct implementation would be clearer.

LangChain must not hide the entire RAG system from the learner.

## 6.3 Hugging Face / Sentence Transformers

Use local embedding models for documents and queries. Consider embedding latency, resource usage, model characteristics, and hardware constraints. Do not assume the largest model is automatically best.

## 6.4 ChromaDB

Use ChromaDB initially for vector storage, metadata, similarity search, and retrieval experiments. Do not add FAISS or PGVector without a specific, justified experiment.

## 6.5 Ollama

Use Ollama as the local LLM runtime with a practical LLaMA-family model appropriate for the available hardware.

Select the exact model during implementation. Keep model configuration separate from the core architecture so that changing the model does not require rewriting the RAG system.

## 6.6 FastAPI

Provide HTTP endpoints, request validation, response schemas, error handling, health/status checks, RAG functionality, and relevant performance information. Routers must remain thin and delegate application behavior to the application layer.

## 6.7 React

Build a deliberately small interface for asking questions, displaying answers and retrieved context, viewing useful performance metrics, and inspecting selected experiment results.

The frontend is a supporting interface, not a separate major engineering project.

## 6.8 Testing and Quality Tools

- **pytest:** unit, integration, API, workflow, optimization, and regression tests.
- **Ruff:** linting, formatting, and code-quality checks.
- **MyPy:** static type checking.
- **Pyright:** additional type analysis.
- **pre-commit:** automated checks before commits.

Configure these tools in Phase 1 and refine them as requirements evolve.

## 6.9 Git and GitHub

Use Git from the beginning and GitHub for version history, documentation, development milestones, and portfolio visibility. Commits must be meaningful, logically separated, and traceable.

## 6.10 Technology Selection Principle

Select technology based on project relevance, learning value, simplicity, maintainability, local compatibility, and measurable benefit. Do not add technologies merely because they are popular.

---

# 7. Project Roadmap

The project progresses through eleven phases. Each phase combines learning, implementation, validation, and Git/GitHub progression.

**Core sequencing:** establish the foundation, build an OOP-based baseline, strengthen engineering quality, instrument and measure, analyze bottlenecks, perform justified optimization experiments, expose the system through an API and UI, validate results, and finalize the portfolio.

## Phase 1 — Project Foundation

### Objective

Establish the repository, development environment, initial project structure, tooling, and conventions without prematurely implementing the RAG system.

### Main Work

- Create and configure the GitHub repository and local Git repository.
- Establish the Python project and initial directory structure.
- Configure `.gitignore`, Ruff, MyPy, Pyright, pytest, and pre-commit.
- Establish configuration management and basic project conventions.
- Create the initial README and first meaningful commit.
- Push the initial project to GitHub.

### Architectural and Learning Focus

Understand project organization, dependency management, configuration, Git/GitHub, static analysis, testing foundations, and modular architecture.

Define initial boundaries without prematurely designing the entire system. Potential areas include `domain/`, `application/`, `rag/`, `embeddings/`, `llm/`, `optimization/`, `observability/`, `api/`, and `tests/`. Confirm the structure before implementation.

### Validation

- The project can be installed and executed locally.
- Tests and configured quality tools run successfully.
- Git status is clean and GitHub is synchronized.
- The user understands the purpose of the main directories and tools.

---

## Phase 2 — Baseline RAG

### Objective

Build a functional Naive RAG system using **OOP from the beginning**. Establish a reliable baseline before introducing performance optimizations.

### Stage 1 — Architecture and Design

- Define the system architecture and end-to-end data flow.
- Define class responsibilities, relationships, and module boundaries.
- Establish appropriate encapsulation, composition, interfaces, and dependency injection where justified.
- Define dependency direction and external-system boundaries.
- Define testing and acceptance criteria.
- Document and approve the architecture before implementation.

### Stage 2 — OOP Component Implementation

- Implement the approved component design incrementally.
- Build document loading, chunking, embedding generation, vector storage, retrieval, context construction, prompt construction, and generation.
- Keep classes cohesive, with clear responsibilities and explicit dependencies.
- Integrate components into a functional end-to-end RAG pipeline.
- Use LangChain meaningfully while understanding the underlying operations.
- Add focused tests as components are implemented.

### Stage 3 — Functional Validation

Verify that the system:

- Ingests the selected documents.
- Generates embeddings and stores vectors in ChromaDB.
- Retrieves relevant information.
- Constructs context and generates answers through Ollama.
- Returns useful retrieval information.
- Handles expected failure cases.
- Produces repeatable results for the same configuration.

Create a small initial set of representative questions.

**Constraint:** Do not introduce optimizations merely because they sound useful. Keep the baseline deliberately simple.

### Git Milestones

Examples: `feat: add document ingestion`, `feat: add embedding pipeline`, `feat: add Chroma vector store`, `feat: implement baseline RAG pipeline`, and `test: add baseline RAG tests`.

---

## Phase 3 — Professional Engineering and Testing

### Objective

Review and strengthen the baseline architecture without treating OOP as something to add for the first time. The objective is to validate and improve the object-oriented design established in Phase 2.

### Main Work

- Review class responsibilities, cohesion, coupling, module boundaries, and dependency direction.
- Apply Clean Code and SOLID where they solve concrete problems.
- Refactor when justified by readability, testability, or maintainability.
- Introduce interfaces, protocols, or dependency injection when they provide meaningful benefits.
- Isolate external dependencies appropriately in unit tests.
- Add unit, integration, retrieval, and RAG workflow tests.

### Validation

The system retains baseline functionality, has understandable class responsibilities, passes the agreed tests and quality checks, and supports isolated testing of appropriate components.

The user must be able to explain the important design decisions and why each abstraction exists.

---

## Phase 4 — Observability and Measurement

### Objective

Instrument the RAG system sufficiently to understand its behavior before optimization.

### Main Work

Introduce useful request identification, timing instrumentation, retrieval metrics, context-size information, token-related metrics where available, LLM timing, TTFT measurement where practical, total latency, error information, and relevant model/configuration details.

Potential measurement structure:

```text
Request
 ├── request_id
 ├── timestamp
 └── configuration/version

Retrieval
 ├── retrieval_latency
 ├── k
 └── documents_retrieved

Context
 ├── chunks_used
 └── context_size

LLM
 ├── model
 ├── TTFT
 ├── generation_latency
 └── token information

Request
 └── total_latency
```

Implement only metrics that answer useful engineering questions.

### Validation

The measurements should help identify retrieval cost, generation cost, context size, token usage where available, end-to-end latency, and obvious performance bottlenecks.

---

## Phase 5 — Baseline Performance Analysis

### Objective

Analyze the baseline and identify actual bottlenecks before implementing optimizations. This is primarily an evidence-based decision phase.

### Main Work

- Examine retrieval, embedding, vector-search, context, token, generation, TTFT, latency, and resource measurements as relevant.
- Distinguish symptoms from bottlenecks.
- Document the baseline configuration, workload, measurements, observed behavior, and limitations.
- Form and prioritize optimization hypotheses.
- Identify components that do not currently justify optimization.

### Validation

Produce a documented baseline and a reasoned selection of experiments for Phases 6 and 7.

**Critical rule: Do not optimize what has not been measured.**

---

## Phase 6 — LLM and Generation Optimization

### Objective

Investigate selected generation-side optimizations justified by baseline evidence.

### Candidate Experiments

- Token usage and context-size reduction.
- Context compression.
- Response caching.
- Streaming and TTFT.
- Model comparison or selection.
- Generation configuration.
- Model routing, if justified.

### Experiment Method

**Baseline → Hypothesis → Controlled Change → Measurement → Quality Validation → Comparison → Keep or Reject**

For example, test whether reducing context lowers latency or token usage while preserving the information needed to answer questions correctly.

### Validation

Each selected experiment must have a defined hypothesis, comparable measurements, quality/regression checks, documented results, and a final decision. A faster answer that loses important information is not automatically an improvement.

---

## Phase 7 — RAG Optimization

### Objective

Investigate evidence-supported optimization opportunities in retrieval and the broader RAG pipeline.

### Candidate Experiments

- Retrieval `k` optimization.
- Chunking efficiency.
- Retrieval latency.
- Embedding efficiency and model comparison.
- Vector-search configuration.
- Retrieval/context efficiency.
- End-to-end RAG latency.

### Quality and Validation

Verify that relevant information remains retrievable, important chunks are not systematically excluded, answers remain grounded, and important questions do not regress.

Prioritize experiments according to baseline evidence, expected benefit, learning value, and implementation cost. Follow the same controlled methodology used in Phase 6.

---

## Phase 8 — FastAPI Backend

### Objective

Expose the RAG system through a clean API boundary.

### Main Work

- Implement appropriate HTTP endpoints.
- Define request/response schemas and validation.
- Handle expected API errors.
- Provide health/status information.
- Expose core RAG functionality and selected performance/experiment information where appropriate.
- Add automated API tests.

Potential endpoints include question/answer, ingestion, health/status, and selected performance information. Define the exact API surface during implementation.

### Architectural Principle

```text
HTTP Request
    ↓
Router
    ↓
Application Use Case
    ↓
RAG System
    ↓
Result
    ↓
Response Schema
```

Keep routers thin and prevent infrastructure details from leaking unnecessarily into the API layer.

---

## Phase 9 — React UI and Performance Dashboard

### Objective

Create a small interface that makes the RAG system and optimization results visible.

### Main Capabilities

- Enter questions and display answers.
- Inspect retrieved context or citations.
- Display useful performance information.
- View selected experiment results and baseline comparisons.

Potential metrics include total latency, retrieval latency, generation latency, TTFT, token information, and configuration.

### Validation

The interface communicates correctly with FastAPI and displays answers, retrieval information, and selected metrics reliably.

Keep React lightweight. If frontend work becomes disproportionate, reduce the UI scope rather than expanding the project unnecessarily.

---

## Phase 10 — Validation and Regression

### Objective

Determine whether the optimized system improves on the baseline without unacceptable regressions.

### Golden Dataset

Create a controlled set of approximately 20–30 representative questions covering:

- Direct factual questions.
- Single-document questions.
- Questions requiring multiple chunks.
- Similar or competing information.
- Questions whose answers are absent from the corpus.
- Retrieval-sensitive questions.

Each case should define expected information and behavior rather than requiring an exact generated sentence.

### Validation Dimensions

- **Functional:** the system works and generates answers.
- **Retrieval:** relevant documents and chunks remain retrievable.
- **Grounding:** answers are supported by retrieved information.
- **Out-of-corpus behavior:** unsupported information is not confidently invented.
- **Performance:** relevant latency, token, context, and resource measurements.
- **Regression:** changes do not introduce unacceptable deterioration.

### Evaluation Boundary

Use a small, controlled, understandable validation approach. Do not build a full evaluation platform or introduce RAGAS and LLM-as-a-judge infrastructure.

### Final Decision

Consider performance, resource usage, useful behavior, and regression risk together. Classify important changes as accepted, rejected, inconclusive, or neutral.

---

## Phase 11 — Portfolio Finalization

### Objective

Present a reproducible, understandable, technically defensible portfolio project.

### Documentation

Document:

- Project objective, architecture, and technology stack.
- Setup, configuration, and usage.
- OOP design and significant architectural decisions.
- Baseline RAG and measurement methodology.
- Baseline performance and observed bottlenecks.
- Optimization hypotheses, experiments, results, and trade-offs.
- Rejected experiments and their rationale.
- Testing, regression methodology, and limitations.
- Final architecture and future improvements.

### Final Technical Review

Review architecture, class responsibilities, code quality, tests, type safety, observability, experiments, regression results, API, UI, documentation, Git history, and reproducibility.

The final repository should be understandable to another engineer without access to the original development conversation.

### Portfolio Narrative

**Build Baseline → Instrument → Measure → Identify Bottlenecks → Design Experiments → Optimize → Validate → Document Trade-offs**

The portfolio should emphasize engineering reasoning rather than the number of technologies used.

## 7.12 Roadmap Progression Principle

Phases follow learning dependencies but are not rigidly isolated. Later evidence may justify revisiting earlier decisions.

For example, baseline analysis may reveal an architectural problem that requires refactoring before optimization continues. Such refactoring is part of professional engineering, not a failure.

## 7.13 Phase Completion Principle

A phase is complete when both conditions are sufficiently validated:

- **Technical outcome:** the intended capability works and has been tested.
- **Learning outcome:** the user understands what was implemented, why it exists, how it works, and its trade-offs.

**“The code runs” is not sufficient for phase completion.**

---

# 8. Global Testing and Quality Strategy

## 8.1 Objective

Testing and quality are first-class concerns throughout the project, not tasks postponed until the end.

Validation must provide confidence in functional behavior, architectural boundaries, optimization experiments, and regression resistance. The goal is useful confidence, not maximum test coverage.

## 8.2 Testing Layers

Introduce testing progressively:

**Unit Tests → Integration Tests → RAG Workflow Tests → API Tests → Performance Experiments → Regression Validation**

Each layer serves a different purpose.

## 8.3 Unit Tests

Test isolated behavior such as configuration handling, text processing, chunking-related logic, context and prompt construction, metric calculations, cache behavior, and application-level decisions.

Avoid unnecessary dependence on ChromaDB, Ollama, filesystem state, or network services when behavior can be tested independently.

OOP design should support testing through cohesive responsibilities and replaceable dependencies where justified.

## 8.4 Integration Tests

Validate interactions between components, including embedding integration, ChromaDB, retrieval, Ollama, RAG execution, and persistence.

Use real local infrastructure when it provides meaningful confidence. Distinguish isolated unit behavior from real component interactions.

## 8.5 RAG Workflow Tests

Validate the primary flow:

**Question → Retrieval → Context → Prompt → LLM → Answer**

Tests should verify that components remain correctly connected without assuming that a generative model always produces identical wording.

## 8.6 API Tests

Once FastAPI is introduced, test request validation, successful requests, expected errors, response schemas, health endpoints, and relevant application behavior.

Validate the API contract without unnecessarily duplicating all lower-level tests.

## 8.7 Performance Experiments

Performance measurements are controlled experiments, not ordinary unit tests.

Each significant experiment must document:

- Baseline and experimental configuration.
- Workload and experimental conditions.
- Hypothesis and controlled change.
- Measured metrics.
- Quality checks.
- Results, interpretation, and decision.

## 8.8 Golden Dataset and Quality Validation

Maintain approximately 20–30 representative questions covering direct facts, single-document and multi-chunk questions, similar information, absent information, and retrieval-sensitive cases.

For each relevant case, record the question, expected information, relevant document/chunk, and expected behavior.

Evaluate retrieval relevance, grounding, expected answer behavior, and appropriate handling of out-of-corpus questions.

## 8.9 Regression and Experiment Decisions

Compare the optimized system against the baseline or previous accepted version across:

- Performance and resource usage.
- Retrieval behavior.
- Grounding and answer behavior.

Classify changes as:

- **Accepted:** useful improvement without unacceptable regression.
- **Rejected:** unacceptable deterioration or unjustified complexity.
- **Inconclusive:** insufficient or inconsistent evidence.
- **Neutral:** insufficient benefit to justify the change.

Not every experiment belongs in the final architecture.

## 8.10 Static Checks and Manual Review

Use Ruff, MyPy, Pyright, pytest, and pre-commit consistently. Establish initial configuration in Phase 1 and refine it as necessary.

Manual review must also consider readability, naming, class and module responsibilities, duplication, dependency direction, justified abstractions, error handling, test usefulness, configuration, logging, and coupling.

## 8.11 Testing Principle

> Test behavior and boundaries, not implementation details unnecessarily.

Refactoring that preserves behavior should not require rewriting tests merely because internal implementation details changed.

---

# 9. Git and GitHub Strategy

## 9.1 Objective and Repository

Repository: `rag-performance-optimization-system`

Git and GitHub are part of the engineering workflow from the beginning. The repository should contain the source, tests, configuration, and documentation required to reproduce the system.

Never commit secrets, credentials, unnecessary generated files, virtual environments, local caches, or machine-specific configuration that should remain local.

## 9.2 Development Workflow

**Understand → Design → Implement Small Change → Run → Test → Review → Quality Checks → Commit → Push**

This workflow reinforces learning, code quality, and controlled progression.

## 9.3 Commit Strategy

Commits must represent meaningful, logically coherent changes.

Examples:

- `chore: initialize project structure`
- `chore: configure development tooling`
- `feat: add document ingestion pipeline`
- `feat: add Chroma vector store integration`
- `feat: implement baseline RAG pipeline`
- `test: add retrieval integration tests`
- `refactor: clarify retrieval component responsibilities`
- `feat: add request performance instrumentation`
- `test: validate cache behavior`
- `docs: document optimization experiment results`

Avoid vague messages such as `stuff`, `changes`, `final`, or `misc`.

Aim for one coherent engineering change per commit without artificially fragmenting related implementation and tests.

## 9.4 Branching and History

Use a simple workflow appropriate for a single-developer portfolio project. Feature branches are optional when they help isolate substantial experiments, refactors, or risky changes. Complex GitFlow is unnecessary.

The history should tell a clear story:

**Foundation → Baseline RAG → Engineering and Testing → Observability → Baseline Analysis → Optimization → API → UI → Validation → Documentation**

Significant optimization experiments should be traceable to their baseline, changes, measurements, and decisions.

## 9.5 Git Hygiene and Learning

Before committing, verify that there are no secrets, unnecessary local files, virtual environments, caches, temporary datasets, debugging artifacts, or inappropriate machine-specific settings.

By project completion, the user should understand repository initialization, staging, commits, branches when justified, merges, remote repositories, push/pull, history review, and safe correction of mistakes.

**Git is part of the learning objective, not merely a way to upload code.**

---

# 10. Definition of Done

The project is complete when it demonstrates a working, measurable, tested, optimized, and professionally engineered RAG system, together with a clear understanding of the decisions behind it.

Completion does not require implementing every possible optimization.

## 10.1 Functional Completion

The system can ingest the selected corpus, chunk documents, generate embeddings, store vectors in ChromaDB, retrieve relevant information, construct context, generate answers through Ollama, return useful retrieval information, and handle expected failure cases.

The baseline must be reproducible using documented configuration.

## 10.2 Architecture and OOP Completion

The architecture demonstrates:

- Clear class and module responsibilities.
- Cohesive components with appropriate encapsulation.
- Appropriate composition and class relationships.
- Reasonable dependency direction.
- Interfaces, protocols, and dependency injection where justified.
- Separation between application logic and infrastructure.
- Separation between API and application logic.
- Maintainable configuration and replaceable dependencies where useful.

The architecture must remain understandable. Unjustified complexity should be removed.

## 10.3 Software Engineering Completion

Demonstrate pragmatic application of Clean Code, SOLID, OOP, separation of concerns, high cohesion, low coupling, type hints, dependency and configuration management, error handling, logging, testability, and modular design.

The user should be able to explain where these principles are applied and where a simpler solution was intentionally preferred.

## 10.4 Testing and Quality Tooling Completion

Provide meaningful unit, integration, RAG workflow, API, and regression validation.

The agreed quality checks using Ruff, MyPy, Pyright, pytest, and pre-commit must work and pass. Document and justify any intentional exceptions.

The number of tests is not itself a completion criterion; important behaviors and boundaries must have meaningful validation.

## 10.5 Observability and Baseline Completion

The system must measure relevant aspects of retrieval latency, context size, token usage where available, LLM latency, TTFT where practical, total latency, errors, and configuration/model information.

Document the baseline configuration, workload or evaluation set, measurements, retrieval and answer behavior, observed bottlenecks, and limitations.

## 10.6 Optimization and Experiment Completion

Investigate a meaningful selection of optimizations in both areas, based on evidence:

- **LLM/generation:** context/token optimization, caching, streaming, generation configuration, or model selection.
- **RAG/retrieval:** retrieval `k`, chunking, retrieval efficiency, embedding efficiency, or vector-search behavior.

Only experiments justified by the baseline need to be implemented.

Document each significant experiment's hypothesis, baseline, change, measurements, quality validation, results, decision, and trade-offs. Rejected experiments remain valuable engineering evidence.

## 10.7 Quality and Regression Completion

Evaluate the final optimized system against the golden dataset. Demonstrate that selected optimizations do not introduce unacceptable regressions in retrieval, grounding, expected answer behavior, or out-of-corpus behavior.

Interpret performance gains together with useful behavior and complexity.

## 10.8 API and UI Completion

FastAPI must provide a clean, tested interface with request validation, response schemas, predictable errors, health/status capability, and access to core RAG functionality.

React must provide a lightweight interface for asking questions, displaying answers, inspecting retrieved context, and viewing relevant performance or experiment information.

## 10.9 Documentation and Git Completion

Document the objective, architecture, technology stack, setup, configuration, usage, baseline methodology and results, OOP design, optimization experiments, trade-offs, validation, limitations, and future work.

The repository should have meaningful commits, clean history, no secrets or unnecessary artifacts, and a reproducible setup. Another engineer should be able to understand the project without access to the original development conversation.

## 10.10 Learning Completion

The user must be able to explain:

**RAG:** the complete pipeline, chunking, embeddings, vector retrieval, context construction, and LangChain's role.

**Python and OOP:** important modules and classes, types, class responsibilities, dependencies, encapsulation, composition, and the design choices that support testability.

**Software Engineering:** why the architecture is structured as it is, where SOLID is applied, why abstractions exist, and why unnecessary abstractions were rejected.

**Performance:** baseline bottlenecks, measurement methods, experiment design, optimization results, and trade-offs.

**Testing:** what unit and integration tests validate and how regression and optimization quality are assessed.

**Engineering Decisions:** the ability to defend significant architectural and optimization choices in a technical interview.

---

# 11. Project Risks and Scope-Control Rules

## 11.1 Overengineering and Premature Abstraction

The project combines RAG, LangChain, OOP, optimization, observability, FastAPI, React, and testing. The main risk is disproportionate complexity.

Do not create abstractions simply because SOLID recommends interfaces, enterprise architectures use factories, dependency injection frameworks exist, or future requirements might change.

A class, interface, or abstraction must solve a concrete problem, such as improving testability, isolating infrastructure, clarifying responsibilities, enabling a justified experiment, or making a real change easier.

**Do not maximize the number of classes. Maximize justified design quality.**

## 11.2 Scope Creep and Technology Collection

A proposed feature must meet at least one condition:

1. It directly supports a project objective.
2. It provides significant learning value.
3. It is necessary to validate an important hypothesis.
4. It solves a concrete implementation problem.

Otherwise, classify it as optional or future work.

Do not add LangGraph, agent frameworks, GraphRAG, hybrid search, rerankers, multiple vector databases, unnecessary observability platforms, complex evaluation frameworks, cloud infrastructure, or Kubernetes merely for portfolio keywords.

## 11.3 Baseline-First Rule

No significant optimization should be introduced just because it is considered a best practice.

**Baseline → Measure → Identify Problem → Form Hypothesis → Optimize → Measure Again → Validate Quality**

Do not optimize a component that has not been understood and measured.

## 11.4 OOP and LangChain Scope Rules

Apply OOP where it improves encapsulation, cohesion, replaceability, testability, or clarity. Keep simple operations as functions when that is the better design.

Use LangChain meaningfully without allowing it to obscure the underlying RAG architecture. For important abstractions, understand what LangChain provides, what happens underneath, and when a direct implementation may be clearer.

## 11.5 Quality-versus-Performance Rule

A faster system is not automatically a better system.

Evaluate:

**Latency + Resource Usage + Retrieval Quality + Answer Quality + System Complexity**

A small performance improvement may not justify significant quality loss or disproportionate complexity.

## 11.6 Measurement and Observability Rules

Performance results depend on hardware, RAM, GPU availability, operating system, model, configuration, corpus, workload, and concurrent activity.

Document relevant conditions and treat measurements as experimental results for the defined environment, not universal benchmarks.

Do not measure everything simply because measurement is possible. For each metric, ask:

> What engineering decision will this metric help us make?

If there is no meaningful answer, the metric may not be necessary.

## 11.7 Experiment and Reproducibility Rules

Candidate experiments are not commitments. Select experiments based on evidence, learning value, expected impact, and implementation cost.

Record relevant model and configuration, retrieval and chunking settings, corpus, hardware/environment, workload, and optimization changes whenever practical.

Classify new ideas as required, useful but optional, future work, or out of scope before expanding the current phase.

## 11.8 Learning-Ownership Rule

The assistant supports implementation but does not replace the user's understanding.

**Understand → Design Together → Focused Guidance → User Implements → Run → Test → Review → Refine**

Avoid large sections of opaque code. The user must own the implementation and understand its important parts.

## 11.9 React Scope Rule

The frontend remains small. If it begins consuming disproportionate time, reduce its scope instead of allowing it to become a second major project.

The primary learning value remains RAG engineering, Python/OOP, performance engineering, optimization, and testing.

## 11.10 Final Scope-Control Principle

> **Optimize for depth of understanding and engineering quality, not breadth of technologies.**

A smaller system that the user can explain, measure, test, optimize, and defend technically is more valuable than a larger system whose architecture and behavior are only partially understood.

