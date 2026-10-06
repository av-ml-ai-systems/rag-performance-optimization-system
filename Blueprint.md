# RAG Performance Optimization System

## Project Roadmap & Learning Blueprint

---

# 1. Project Definition

## 1.1 Project Name

**RAG Performance Optimization System**

## 1.2 Repository Name

`rag-performance-optimization-system`

## 1.3 Project Type

A practical AI/ML engineering project focused on building, measuring, analyzing, and optimizing a modular Retrieval-Augmented Generation (RAG) system.

The project combines:

* RAG engineering
* Python software engineering
* Performance engineering
* Observability and measurement
* Controlled optimization experiments
* Testing and validation
* API development
* Lightweight frontend development

## 1.4 Central Engineering Question

> **Can we build a clean, modular RAG system and systematically measure and optimize its performance and resource usage without unnecessarily sacrificing useful behavior?**

## 1.5 Core Idea

The project will begin with a deliberately simple **Naive RAG system**.

We will first establish a reliable baseline and then use measurement and controlled experiments to identify meaningful optimization opportunities.

The project will follow this general engineering loop:

**Build → Measure → Understand → Identify bottleneck → Form hypothesis → Optimize → Measure again → Validate quality → Keep or reject the optimization**

The objective is **not simply to make the system faster or cheaper**.

An optimization is considered successful only when its performance or resource benefit is meaningful while useful system behavior is preserved.

---

# 2. Project Objectives

The project has four major objectives.

## 2.1 RAG Engineering Objective

Build a modular RAG application using:

* LangChain
* Hugging Face / Sentence Transformers
* ChromaDB
* Ollama
* A practical LLaMA-family local model

The system should support:

* Document ingestion
* Document chunking
* Embedding generation
* Vector storage
* Retrieval
* Context construction
* Prompt construction
* LLM generation
* Returning the answer together with useful retrieval information

The purpose is not to build the most sophisticated RAG architecture possible.

The purpose is to establish a **controlled and understandable RAG system** that can subsequently be measured and optimized.

## 2.2 Performance Optimization Objective

Systematically investigate relevant performance and resource dimensions.

### Generation / LLM-side optimization

* Token usage
* Context size
* Caching
* Streaming
* Time to First Token (TTFT)
* Generation latency
* Resource efficiency
* Model selection or routing when justified

### RAG / Retrieval-side optimization

* Chunking efficiency
* Retrieval parameters
* Embedding efficiency
* Vector search
* Retrieval latency
* End-to-end RAG latency
* Embedding resource usage

The project will **not assume that every possible optimization must be implemented**.

Baseline measurements will determine which optimization experiments are relevant and valuable.

## 2.3 Software Engineering Objective

Develop the project using professional Python software engineering practices.

This is a **first-class objective of the project**, not merely a supporting concern.

The implementation should emphasize:

* Clean Code
* SOLID principles
* Separation of concerns
* Modular architecture
* High cohesion and low coupling
* Dependency inversion
* Dependency injection where justified
* Appropriate use of interfaces and protocols
* Type hints
* Clear naming
* Small and focused functions
* Small and cohesive modules
* Explicit configuration
* Appropriate exception handling
* Testability
* Maintainability
* Reusability where genuinely justified
* DRY where appropriate
* KISS
* Avoidance of premature abstraction
* Avoidance of unnecessary complexity

The project should also use professional development and quality practices, including:

* Git and GitHub
* pytest
* Ruff
* MyPy
* Pyright
* pre-commit
* Automated quality checks
* Meaningful commits
* Clear documentation

### Engineering Principle

> **The goal is not to make the code as abstract or sophisticated as possible. The goal is to make it clean, understandable, maintainable, testable, and appropriately designed for the actual problem.**

## 2.4 Learning Objective

By the end of the project, the user should be able to:

* Explain the complete RAG architecture
* Explain the role of each major technology
* Understand the important Python code written during the project
* Explain why important abstractions exist
* Explain why unnecessary abstractions were avoided
* Apply Clean Code principles in a real project
* Apply SOLID principles appropriately rather than mechanically
* Design modular Python components
* Write and understand automated tests
* Measure system performance
* Identify performance bottlenecks
* Design controlled optimization experiments
* Interpret optimization results
* Understand performance/quality trade-offs
* Defend architectural and engineering decisions in a technical interview

The project should therefore produce both:

**A working system**
and
**a strong understanding of how and why the system was built.**

---

# 3. Scope & Boundaries

The scope of the project must remain controlled throughout development.

## 3.1 Core Project Scope

The project must include:

* A working Naive RAG system
* Document ingestion
* Chunking
* Embeddings
* ChromaDB vector storage
* Retrieval
* LLM generation through Ollama
* LangChain as a significant practical component
* Hugging Face / Sentence Transformers
* Professional Python architecture
* Software engineering practices
* Automated testing
* Performance measurement
* Baseline performance analysis
* Selected LLM/generation optimization experiments
* Selected RAG optimization experiments
* FastAPI backend
* Small React interface
* Lightweight performance/experiment information
* Regression and quality validation
* Git/GitHub workflow
* Professional project documentation

## 3.2 Software Engineering Scope

Software engineering is an explicit part of the project scope.

The implementation should apply, where appropriate:

* Clean Architecture principles
* SOLID principles
* Clean Code practices
* Separation of concerns
* Dependency inversion
* Dependency injection
* Modular design
* Type safety
* Unit testing
* Integration testing
* API testing
* Static analysis
* Linting and formatting
* Pre-commit quality checks
* Configuration management
* Error handling
* Logging and observability
* Maintainable Git history

These practices should be introduced **progressively as the project evolves**, rather than attempting to design the entire architecture before the actual requirements are understood.

## 3.3 Conditional / Experimental Scope

The following are possible experiments or improvements but are **not mandatory from the beginning**:

* Response caching
* Streaming
* Model comparison
* Model routing
* Advanced context compression
* Embedding model comparison
* Retrieval parameter optimization
* Additional retrieval experiments
* Additional performance instrumentation

Their inclusion will depend on the baseline measurements, project scope, available time, and whether they provide meaningful learning or optimization value.

## 3.4 Out of Scope

The following will not be part of the core project:

* Agentic RAG
* GraphRAG
* Multi-agent systems
* LangGraph
* Hybrid search
* Complex query rewriting
* Multi-query retrieval
* Reranking unless a strong project-specific justification emerges
* Fine-tuning
* Multimodal RAG
* Cloud deployment
* Kubernetes
* Distributed systems
* Enterprise-scale architecture
* Full evaluation platforms
* RAGAS
* LLM-as-a-judge infrastructure
* Comprehensive guardrail systems
* LangSmith as a required dependency

The project is about **RAG performance engineering and optimization**, not about implementing every modern RAG technique.

---

# 4. Learning Strategy

## 4.1 Core Learning Principle

> **The user writes and owns the Python code.**

The implementation process will prioritize understanding over speed.

ChatGPT will provide:

* Explanations
* Architectural reasoning
* Small focused code examples
* Implementation guidance
* Code review
* Debugging assistance
* Testing guidance
* Engineering feedback

Large blocks of generated implementation code will be avoided.

## 4.2 Implementation Learning Loop

Each meaningful implementation step will follow this process:

**Understand → Design → Small implementation step → User writes/assembles the code → Run → Test → Review → Fix/understand → Quality checks → Commit**

Only after the current step is understood and validated will the project move forward.

## 4.3 Python Learning Principles

Throughout the project, particular attention will be given to:

* Writing readable Python
* Understanding Python language features rather than copying patterns
* Type hints
* Functions and classes
* Modules and packages
* Exceptions
* Interfaces and protocols
* Dependency injection
* Composition
* Testing
* Configuration
* File and project organization
* Maintainability
* Refactoring

When a Python construct is unfamiliar, it will be explained before becoming part of the implementation.

## 4.4 Software Engineering Learning Principles

The project will be used to practice software engineering principles in a real context.

The emphasis will be on understanding:

* **Why** a design decision is necessary
* **What problem** an abstraction solves
* **What complexity** it introduces
* **When** a principle should be applied
* **When not** to apply it
* How different design decisions affect maintainability and testability

SOLID and Clean Code will therefore be treated as **engineering tools**, not rules to apply mechanically.

## 4.5 Scope-Control Principle

The project should not become increasingly complex simply because additional technologies or techniques are available.

The default question for any proposed addition should be:

> **Does this directly support the project's objective or provide significant learning value without creating disproportionate complexity?**

If the answer is no, the feature should remain outside the project.

## 4.6 Validation Before Progression

Each phase should have explicit validation criteria.

We should not move to the next major phase simply because the code "runs."

Validation may include:

* Functional tests
* Unit tests
* Integration tests
* Quality checks
* Performance measurements
* Retrieval checks
* Regression checks
* Manual inspection
* Architectural review
* Understanding of the implemented code

A phase is complete when its **technical outcome and learning outcome** have been sufficiently validated.

# 5. Technical Architecture

## 5.1 Architectural Objective

The architecture should provide a clean separation between:

* User interface
* API layer
* Application/use-case logic
* Domain concepts
* RAG components
* Infrastructure implementations
* Observability
* Optimization mechanisms

The architecture should remain **simple enough to understand and evolve**, while providing appropriate boundaries for testing and optimization.

The project should avoid both extremes:

* A monolithic implementation where everything is coupled together
* An unnecessarily complex enterprise architecture with abstractions that the project does not need

---

## 5.2 High-Level Architecture

The initial architectural direction is:

```text
┌──────────────────────┐
│      React UI        │
└──────────┬───────────┘
           │ HTTP
           ▼
┌──────────────────────┐
│      FastAPI         │
│      API Layer       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Application Layer    │
│      Use Cases       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Domain / Contracts │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────┐
│       Infrastructure / RAG      │
│                                 │
│ LangChain                       │
│ ChromaDB                        │
│ Hugging Face                    │
│ Ollama                          │
│ Cache                           │
└─────────────────────────────────┘
```

This is an initial architectural model. The exact module and package structure will be refined during implementation.

---

## 5.3 RAG Data Flow

The core RAG pipeline will initially follow:

```text
Documents
    ↓
Document Loading
    ↓
Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Retriever
    ↓
Retrieved Context
    ↓
Prompt Construction
    ↓
Ollama / LLaMA Model
    ↓
Answer
```

The baseline should remain intentionally simple.

This allows us to establish a controlled reference system before introducing optimization techniques.

---

## 5.4 Application Flow

A typical user request should follow a structure similar to:

```text
User Question
      ↓
FastAPI Endpoint
      ↓
Application Use Case
      ↓
RAG Pipeline
      ↓
Retriever
      ↓
Context Construction
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

FastAPI route handlers should remain thin.

They should coordinate the request and delegate the actual application behavior to the appropriate application/use-case components.

---

## 5.5 Architectural Boundaries

The project should maintain clear boundaries between major responsibilities.

### API Layer

Responsible for:

* HTTP endpoints
* Request validation
* Response schemas
* HTTP-level errors
* API concerns

It should not contain:

* RAG logic
* ChromaDB implementation details
* prompt construction
* optimization logic
* business/application rules

### Application Layer

Responsible for:

* Use cases
* Application workflows
* Coordinating domain and infrastructure components
* Controlling the execution flow

Examples may include:

* AskQuestion
* IngestDocuments
* RunExperiment
* RetrieveExperimentResults

The exact use cases will be determined during implementation.

### Domain Layer

Responsible for concepts and contracts that should not depend unnecessarily on external frameworks.

Possible concepts include:

* Question
* RetrievedDocument
* RetrievedContext
* Answer
* RAGResult
* ExperimentResult

These are preliminary candidates rather than mandatory classes.

### Infrastructure Layer

Responsible for external technologies and implementations, such as:

* LangChain
* ChromaDB
* Hugging Face
* Ollama
* caching mechanisms
* filesystem access
* persistence

The application layer should not need to understand the internal details of these technologies.

---

## 5.6 Dependency Direction

A key architectural principle is to avoid unnecessary coupling toward infrastructure.

The preferred conceptual dependency direction is:

```text
API
 ↓
Application
 ↓
Domain / Contracts
 ↑
Infrastructure Implementations
```

Infrastructure may implement contracts required by the application.

The application should not become tightly coupled to ChromaDB, Ollama, or a particular LangChain implementation merely because those technologies are used internally.

However, dependency inversion should be applied **only where it provides real value**.

We will not create interfaces for every class simply to satisfy a theoretical architecture.

---

## 5.7 Optimization Architecture

Optimization mechanisms should remain separate from the core RAG reasoning whenever practical.

Conceptually:

```text
                  RAG Application
                        │
          ┌─────────────┼─────────────┐
          │             │             │
       Retrieval     Generation    Context
          │             │             │
          └─────────────┼─────────────┘
                        │
                 Measurement
                        │
          ┌─────────────┴─────────────┐
          │                           │
     Observability              Optimization
          │                           │
     latency                    caching
     tokens                     retrieval
     TTFT                       context
     errors                     embeddings
     timing                     streaming
```

This separation allows us to experiment with optimization strategies without rewriting the entire RAG application.

---

## 5.8 Architecture Principle: Baseline First

The baseline architecture should remain as simple as reasonably possible.

We should not introduce:

* caching
* routing
* complex retrieval strategies
* advanced compression
* elaborate abstraction layers

before we have established whether they are necessary.

The baseline provides the controlled reference against which later changes can be evaluated.

---

## 5.9 Architecture Evolution

The architecture is intentionally evolutionary.

We will follow:

```text
Simple requirement
      ↓
Simple implementation
      ↓
Actual problem discovered
      ↓
Refactor / introduce abstraction
      ↓
Test
      ↓
Continue
```

rather than:

```text
Imagine every possible future requirement
      ↓
Build huge abstraction hierarchy
      ↓
Hope it becomes useful
```

This is an important part of the software-engineering learning objective.

---

# 6. Technology Stack

## 6.1 Core Language

### Python

Python will be the primary implementation language.

It will be used for:

* RAG components
* application logic
* infrastructure integrations
* optimization experiments
* observability
* FastAPI
* testing
* automation

The project will deliberately use Python as a vehicle for strengthening software-engineering skills.

---

## 6.2 RAG Framework

### LangChain

LangChain will be used as a significant practical component of the project.

Potential responsibilities include:

* document loaders
* text splitting
* embeddings integration
* retrievers
* prompt templates
* runnable composition
* model integration
* RAG pipeline composition
* Ollama integration
* other useful RAG utilities

LangChain will **not** replace understanding of the underlying RAG architecture.

For important abstractions, we should understand:

1. What abstraction is being used
2. What problem it solves
3. What complexity it hides
4. Why it is useful in this project
5. When it would not be appropriate

---

## 6.3 Embeddings

### Hugging Face / Sentence Transformers

Hugging Face-based embedding models will be used for document and query embeddings.

The project will consider:

* embedding generation
* embedding latency
* embedding resource usage
* embedding model characteristics
* potential embedding optimization

Model selection will consider the local hardware constraints rather than assuming that the largest available model is automatically better.

---

## 6.4 Vector Database

### ChromaDB

ChromaDB will be the initial vector database.

It will be used for:

* storing document embeddings
* storing document metadata
* similarity search
* retrieval experiments
* measuring retrieval behavior and latency

PGVector and FAISS will not be added unless a specific experiment later provides a strong justification.

---

## 6.5 Local LLM

### Ollama

Ollama will provide the local LLM runtime.

The project will use a practical LLaMA-family model appropriate for the available hardware.

The exact model should be selected during implementation rather than permanently hard-coded into the blueprint.

The application should isolate model configuration so that changing the local model does not require rewriting the RAG architecture.

---

## 6.6 Backend

### FastAPI

FastAPI will provide the application API.

Responsibilities include:

* HTTP endpoints
* request validation
* response schemas
* API error handling
* health checks
* exposing RAG functionality
* exposing relevant performance information

FastAPI routers should remain thin and delegate application behavior to the application layer.

---

## 6.7 Frontend

### React

React will provide a deliberately small user interface.

The interface should focus on:

* asking questions
* displaying answers
* displaying retrieved context/citations
* displaying useful performance information
* viewing selected experiment results

The frontend is **not a separate frontend engineering project**.

Its purpose is to provide a usable interface to the RAG system and make the optimization work visible.

---

## 6.8 Testing

### pytest

pytest will be the primary testing framework.

Testing will include, where appropriate:

* unit tests
* integration tests
* API tests
* RAG workflow tests
* optimization validation tests
* regression tests

The goal is confidence in the system, not maximizing the number of tests.

---

## 6.9 Code Quality

### Ruff

Ruff will be used for Python linting and code-quality checks.

### MyPy

MyPy will be used for static type checking.

### Pyright

Pyright will provide an additional type-analysis perspective and strengthen type-safety practices.

### pre-commit

pre-commit will be used to automate selected quality checks before commits.

---

## 6.10 Version Control

### Git

Git will be used from the beginning of the project.

### GitHub

GitHub will host the project repository and provide:

* version history
* project documentation
* milestones in the development history
* portfolio visibility

Git history should contain small, meaningful, logically separated commits.

---

## 6.11 Technology Selection Principle

Technology will be selected based on:

* relevance to the project objective
* learning value
* simplicity
* maintainability
* compatibility with the local environment
* ability to measure and validate its impact

The project will **not add technologies merely because they are popular in the current AI ecosystem**.

# 7. Project Roadmap

The project will be developed progressively through eleven phases.

The phases are intentionally ordered so that:

* The system is understood before it is optimized.
* The baseline exists before performance decisions are made.
* Software-engineering practices evolve alongside the implementation.
* Observability is established before performance analysis.
* Optimization experiments are driven by evidence rather than assumptions.
* API and UI layers are added after the core system is stable.
* Final validation compares the optimized system against the established baseline.

The roadmap is therefore not simply a list of implementation tasks. Each phase combines:

* Learning objectives
* Engineering objectives
* Implementation work
* Validation
* Git/GitHub progression
* Criteria for moving forward

---

## Phase 1 — Project Foundation

### Objective

Establish the project repository, development environment, initial architecture, engineering tooling, and basic project conventions.

The goal is to create a professional foundation without prematurely implementing the RAG system.

### Main Learning Goals

Understand:

* Python project structure
* Dependency management
* Configuration management
* Git/GitHub workflow
* Static analysis
* Linting and formatting
* Type checking
* Automated testing foundations
* pre-commit
* Basic modular architecture

### Main Implementation Work

* Create the GitHub repository
* Initialize the local Git repository
* Establish the Python project
* Define the initial directory structure
* Establish configuration management
* Configure `.gitignore`
* Configure Ruff
* Configure MyPy
* Configure Pyright
* Configure pytest
* Configure pre-commit
* Create the initial README
* Establish basic project conventions
* Create the first meaningful commit
* Push the initial project to GitHub

### Architectural Focus

Define the initial boundaries without creating unnecessary abstractions.

Potential initial areas include:

* `domain/`
* `application/`
* `rag/`
* `embeddings/`
* `llm/`
* `optimization/`
* `observability/`
* `api/`
* `tests/`

The exact structure will be reviewed before implementation.

### Validation

Before moving forward:

* The project can be installed and executed locally.
* Tests can run successfully.
* Static analysis works.
* Linting works.
* Formatting works.
* Type checking works.
* pre-commit works.
* Git repository is clean.
* GitHub repository is synchronized.
* The user understands the purpose of the main project directories and tools.

---

## Phase 2 — Baseline RAG

### Objective

Build the simplest useful Naive RAG system that can reliably answer questions from a controlled document collection.

This phase establishes the functional baseline.

### Main Learning Goals

Understand the complete RAG lifecycle:

```text
Documents
    ↓
Loading
    ↓
Chunking
    ↓
Embedding
    ↓
Vector Storage
    ↓
Retrieval
    ↓
Context Construction
    ↓
Prompt
    ↓
LLM
    ↓
Answer
```

The user should understand what each component does and why it exists.

### Main Implementation Work

Implement:

* document ingestion
* document loading
* text splitting/chunking
* embedding generation
* ChromaDB persistence
* similarity retrieval
* prompt construction
* Ollama integration
* RAG execution
* answer generation
* retrieval information needed for later analysis

LangChain should be used meaningfully while maintaining understanding of the underlying operations.

### Important Constraint

No performance optimization should be introduced merely because it sounds useful.

The baseline should remain deliberately simple.

### Validation

The baseline must:

* ingest the selected documents
* create embeddings
* store vectors
* retrieve relevant information
* generate answers
* handle basic failure cases
* produce repeatable results for the same configuration

A small initial set of representative questions should be created for functional validation.

### Git Milestones

Examples:

* `feat: add document ingestion`
* `feat: add embedding pipeline`
* `feat: add Chroma vector store`
* `feat: implement baseline RAG pipeline`
* `test: add baseline RAG tests`

---

## Phase 3 — Professional Engineering & Testing

### Objective

Transform the initial working implementation into a clean, modular, testable Python system.

This phase deliberately strengthens software engineering before serious optimization work begins.

### Main Learning Goals

Apply:

* Clean Code
* SOLID
* separation of concerns
* high cohesion
* low coupling
* dependency inversion
* dependency injection where justified
* interfaces/protocols where justified
* type hints
* focused modules
* explicit dependencies
* testable design

### Main Implementation Work

Review and refactor the baseline architecture.

Potential boundaries include:

```text
Domain
   ↓
Application
   ↓
RAG Components
   ↓
Infrastructure
```

Introduce abstractions only where they provide meaningful benefits such as:

* testability
* replaceability
* isolation of external dependencies
* clearer responsibilities

### Testing

Introduce:

* unit tests
* integration tests
* retrieval tests
* RAG workflow tests
* configuration tests where useful

External systems should be isolated or controlled appropriately in unit tests.

### Important Learning Principle

SOLID is not a checklist.

For every abstraction, the project should ask:

> What problem does this abstraction solve?

If the answer is unclear, the abstraction probably should not exist yet.

### Validation

The system should:

* retain baseline functionality
* have clear module responsibilities
* be easier to test
* pass automated tests
* pass quality checks
* maintain understandable dependency relationships
* allow the user to explain the important code and architectural decisions

---

## Phase 4 — Observability & Measurement

### Objective

Introduce the minimum useful instrumentation required to understand how the RAG system behaves.

Optimization cannot be performed responsibly without measurement.

### Main Learning Goals

Understand:

* performance instrumentation
* latency decomposition
* request-level metrics
* token measurement
* timing boundaries
* observability versus excessive logging
* reproducible measurements

### Main Measurements

The initial measurement model should consider:

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
 ├── output_tokens
 └── total_tokens

Request
 └── total_latency
```

Not every metric needs to be implemented immediately. Instrumentation should be proportional to its usefulness.

### Main Implementation Work

Add:

* request identification
* timing instrumentation
* retrieval metrics
* context-size information
* token-related metrics where available
* LLM timing
* TTFT measurement if practical with Ollama streaming
* total request latency
* error information
* relevant configuration information

### Validation

The system should produce enough information to answer:

* How long does retrieval take?
* How long does generation take?
* How large is the retrieved context?
* How many tokens are being processed/generated?
* What is the end-to-end latency?
* Where are the obvious performance costs?

---

## Phase 5 — Baseline Performance Analysis

### Objective

Analyze the baseline and identify the actual performance bottlenecks before implementing optimizations.

This phase is a decision-making phase rather than primarily a coding phase.

### Main Learning Goals

Learn to:

* profile a system
* interpret measurements
* distinguish symptoms from bottlenecks
* formulate optimization hypotheses
* establish a baseline
* avoid optimizing irrelevant components
* reason about trade-offs

### Baseline Analysis Areas

Depending on actual measurements:

* retrieval latency
* embedding generation
* vector search
* context size
* input token usage
* output token usage
* LLM generation latency
* TTFT
* total latency
* resource consumption
* ingestion performance

### Baseline Report

The project should establish a documented baseline containing relevant metrics and configuration.

The baseline should make clear:

* what was measured
* under what configuration
* what the observed behavior was
* what appears to be the bottleneck
* what should be investigated next
* what should **not** be optimized because it is not currently important

### Critical Principle

> Do not optimize what has not been measured.

The baseline determines which Phase 6 and Phase 7 experiments are worth performing.

---

## Phase 6 — LLM / Generation Optimization

### Objective

Investigate selected optimizations on the generation side of the RAG system.

The project will not implement every possible optimization.

Experiments will be selected according to the baseline analysis.

### Candidate Areas

Potential experiments include:

* token usage optimization
* context-size reduction
* context compression
* response caching
* streaming
* TTFT optimization
* model comparison
* model selection
* model routing
* generation configuration

### Candidate Experiment Structure

Each experiment should follow:

```text
Baseline
    ↓
Optimization Hypothesis
    ↓
Controlled Change
    ↓
Measurement
    ↓
Quality Validation
    ↓
Comparison
    ↓
Keep / Reject
```

### Example

A context optimization experiment might ask:

> Can we reduce the amount of context sent to the LLM while preserving useful answer quality?

The experiment would measure both:

* performance/resource impact
* useful behavior

A faster response that loses important information is not automatically an improvement.

### Validation

Each selected optimization should have:

* a clearly defined hypothesis
* a controlled change
* comparable measurements
* documented results
* quality/regression validation
* a final keep/reject decision

---

## Phase 7 — RAG Optimization

### Objective

Investigate optimization opportunities in retrieval and the broader RAG pipeline.

### Candidate Areas

Potential experiments include:

* retrieval `k` optimization
* chunking efficiency
* retrieval latency
* embedding efficiency
* embedding model comparison
* vector-search configuration
* retrieval/context efficiency
* end-to-end RAG latency

### Priority Principle

Likely high-value areas include:

1. Retrieval efficiency
2. Context/token optimization
3. Retrieval latency

Other experiments are conditional on the baseline.

### Quality Consideration

Retrieval optimization must not be evaluated only through latency.

The project should also verify:

* relevant information remains retrievable
* useful documents are not systematically excluded
* answer behavior remains acceptable
* important questions do not regress

### Validation

Each experiment follows the same controlled methodology established in Phase 6.

---

## Phase 8 — FastAPI Backend

### Objective

Expose the RAG system through a clean, production-oriented API boundary.

### Main Learning Goals

Understand:

* FastAPI architecture
* API contracts
* request/response schemas
* validation
* dependency management
* error handling
* separation between API and application logic
* API testing

### Main Implementation Work

Potential endpoints include:

* health/status
* question/answer
* ingestion
* selected performance information
* experiment information where appropriate

The exact API surface will be defined during implementation.

### Architectural Principle

The API layer should remain thin:

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

RAG implementation details should not leak unnecessarily into the HTTP layer.

### Validation

The API should:

* validate requests
* return predictable responses
* handle expected errors
* expose the core RAG functionality
* have automated API tests
* preserve the underlying architecture

---

## Phase 9 — React UI & Performance Dashboard

### Objective

Create a small interface that makes the RAG system and its optimization behavior visible.

### Main UI Capabilities

The interface should allow users to:

* enter questions
* receive answers
* inspect retrieved context
* see relevant performance information
* observe selected optimization results

### Performance Information

The dashboard should remain lightweight.

Potential information includes:

* total latency
* retrieval latency
* generation latency
* TTFT
* token information
* selected configuration
* baseline versus optimized comparison

### Scope Constraint

React should not become a second major project.

The UI exists primarily to:

* interact with the RAG system
* demonstrate the engineering work
* make performance behavior observable

### Validation

The UI should:

* communicate correctly with FastAPI
* display answers correctly
* display useful retrieval information
* display selected performance information
* remain simple and maintainable

---

## Phase 10 — Validation & Regression

### Objective

Determine whether the optimized system actually represents an improvement over the baseline.

This is the final technical validation phase.

### Golden Dataset

Create a small controlled evaluation set of approximately 20–30 representative questions.

Possible categories:

* direct factual questions
* single-document questions
* multi-chunk questions
* similar/competing information
* information absent from the corpus
* retrieval-sensitive questions

Each test case should define useful expected behavior rather than requiring an exact generated sentence.

### Validation Dimensions

The project should compare:

#### Functional Behavior

* Does the system work?
* Are answers generated correctly?
* Are expected documents retrievable?

#### Retrieval Behavior

* Are relevant chunks retrieved?
* Are important questions still supported?

#### Quality / Grounding

* Does the answer remain grounded in the retrieved information?
* Has optimization introduced obvious regressions?

#### Performance

* latency
* retrieval latency
* generation latency
* TTFT where applicable
* context size
* token usage
* relevant resource measurements

### Evaluation Principle

No optimization is considered successful solely because it is faster or cheaper.

The final decision must consider:

```text
Performance
     +
Resource Usage
     +
Useful Behavior
     +
Regression Risk
```

### Evaluation Tooling Boundary

The project will use a small controlled validation approach.

It will not become a full RAG evaluation platform.

Specifically out of scope:

* RAGAS
* LLM-as-a-judge infrastructure
* large-scale evaluation platforms

---

## Phase 11 — Portfolio Finalization

### Objective

Transform the completed project into a clear, reproducible, technically defensible portfolio project.

### Documentation

The final repository should document:

* project objective
* architecture
* technology stack
* setup
* usage
* baseline RAG
* measurement methodology
* baseline performance
* optimization hypotheses
* experiments
* results
* trade-offs
* rejected optimizations
* final architecture
* validation methodology
* limitations
* future improvements

### Portfolio Narrative

The project should communicate a clear engineering story:

```text
Built a baseline RAG system
        ↓
Instrumented the system
        ↓
Measured actual behavior
        ↓
Identified bottlenecks
        ↓
Designed controlled experiments
        ↓
Optimized selected components
        ↓
Validated quality and regression
        ↓
Documented engineering trade-offs
```

The portfolio should emphasize **engineering reasoning**, not merely the number of technologies used.

### Final Technical Review

Before considering the project complete, review:

* architecture
* code quality
* tests
* type safety
* observability
* optimization experiments
* regression results
* API
* UI
* documentation
* Git history
* reproducibility

The final repository should be understandable to another engineer without requiring the original development conversation.

---

## 7.12 Roadmap Progression Principle

The phases are sequential in terms of learning dependencies, but they are not rigidly isolated.

Later phases may reveal problems that require revisiting earlier decisions.

For example:

```text
Phase 5
Baseline Analysis
     ↓
Discover architecture problem
     ↓
Return to Phase 3 principles
     ↓
Refactor
     ↓
Continue optimization
```

This is not considered failure.

Refactoring based on evidence is part of professional software engineering.

The important constraint is to avoid uncontrolled scope expansion.

---

## 7.13 Phase Completion Principle

A phase is complete only when both dimensions have been sufficiently validated:

### Technical Outcome

The intended system capability or engineering improvement works and has been tested.

### Learning Outcome

The user understands:

* what was implemented
* why it was implemented
* how it works
* what trade-offs exist
* why alternative approaches were rejected when relevant

Therefore:

> **"The code runs" is not sufficient as a phase-completion criterion.**

# 8. Global Testing & Quality Strategy

## 8.1 Objective

Testing and code quality are first-class concerns throughout the project.

They will not be postponed until the end.

The objective is to establish enough automated and manual validation to provide confidence that:

* the system behaves correctly
* changes do not introduce unintended regressions
* optimization experiments are trustworthy
* architectural boundaries remain understandable
* the code remains maintainable as the project grows

Testing should remain proportional to the project's complexity.

The goal is **useful confidence**, not maximum test coverage.

---

## 8.2 Testing Strategy

Testing will be introduced progressively.

The general progression is:

```text
Basic Tests
    ↓
Unit Tests
    ↓
Integration Tests
    ↓
RAG Workflow Tests
    ↓
API Tests
    ↓
Performance Validation
    ↓
Regression Validation
```

Each type of test serves a different purpose.

---

## 8.3 Unit Tests

Unit tests will validate isolated pieces of application behavior.

Potential candidates include:

* configuration handling
* text-processing logic
* chunking-related logic
* context construction
* prompt construction
* metric calculations or transformations
* cache behavior
* application-level decision logic

Unit tests should avoid unnecessary dependence on:

* ChromaDB
* Ollama
* filesystem state
* network services

when the behavior can be tested independently.

### Principle

A unit test should make it reasonably clear what failed and why.

---

## 8.4 Integration Tests

Integration tests will validate interactions between important components.

Potential examples include:

* embedding integration
* ChromaDB integration
* retrieval behavior
* Ollama integration
* RAG pipeline execution
* persistence behavior

Integration tests may use real local infrastructure where doing so provides meaningful confidence.

The project should distinguish clearly between:

```text
Unit Test
    ↓
Tests isolated behavior

Integration Test
    ↓
Tests real component interaction
```

---

## 8.5 RAG Workflow Tests

The complete RAG pipeline should have tests covering the primary flow:

```text
Question
   ↓
Retrieval
   ↓
Context
   ↓
Prompt
   ↓
LLM
   ↓
Answer
```

These tests should verify that the major components remain connected correctly.

They should not attempt to prove that a generative model will always produce exactly the same wording.

---

## 8.6 API Tests

Once FastAPI is introduced, automated API tests should validate:

* request validation
* successful requests
* expected error responses
* response structure
* health endpoints
* relevant application behavior

API tests should verify the API contract without duplicating every lower-level unit test.

---

## 8.7 Performance Validation

Performance measurements are not traditional unit tests.

They should instead be treated as controlled experiments.

Each performance experiment should define:

* baseline configuration
* experimental configuration
* measured metrics
* experimental conditions
* relevant quality checks
* interpretation
* final decision

Example structure:

```text
Experiment
    ↓
Baseline
    ↓
Hypothesis
    ↓
Controlled Change
    ↓
Measurement
    ↓
Quality Check
    ↓
Comparison
    ↓
Decision
```

---

## 8.8 Golden Dataset

A small golden dataset will be used for regression validation.

Target size:

**approximately 20–30 representative questions**

The dataset should contain different types of retrieval and generation situations.

Possible categories:

* direct factual questions
* questions answered by one document
* questions requiring multiple chunks
* similar or competing information
* questions whose answer is absent from the corpus
* retrieval-sensitive questions

Each case should contain enough information to evaluate expected behavior.

Potential structure:

```text
Question
Expected Information
Relevant Document / Chunk
Expected Behavior
```

The goal is not to construct a perfect benchmark.

The goal is to create a stable reference set for comparing system versions.

---

## 8.9 Quality Validation

For optimization experiments, quality validation should consider:

### Retrieval

* Was relevant information retrieved?
* Were important documents/chunks lost?

### Grounding

* Is the answer supported by the retrieved context?

### Expected Behavior

* Does the answer contain the information expected for the question?

### Out-of-Corpus Behavior

* Does the system avoid confidently inventing information when the corpus does not contain the answer?

The evaluation should remain lightweight and understandable.

---

## 8.10 Regression Strategy

After an optimization, compare the optimized system against the baseline or previous accepted version.

The comparison should consider:

```text
Performance
    +
Resource Usage
    +
Retrieval Behavior
    +
Answer Behavior
```

An optimization may therefore be:

### Accepted

Performance improves without unacceptable regression.

### Rejected

Performance improves but useful behavior deteriorates significantly.

### Inconclusive

Measurements are insufficient or inconsistent.

### Neutral

The change does not provide enough benefit to justify its additional complexity.

This distinction is important because **not every optimization experiment needs to become part of the final architecture**.

---

## 8.11 Static Quality Checks

The project should progressively integrate:

* Ruff
* MyPy
* Pyright
* pytest
* pre-commit

These tools should be run consistently rather than only before the final publication.

The exact configuration will be established during Phase 1 and refined as the project evolves.

---

## 8.12 Manual Review

Automated tools are not sufficient.

Regular manual review should consider:

* readability
* naming
* module responsibilities
* duplicated logic
* unnecessary abstraction
* dependency direction
* error handling
* test usefulness
* configuration clarity
* logging quality
* architectural coupling

The user should be able to explain the important parts of the system.

---

## 8.13 Testing Principle

The project follows:

> **Test behavior and boundaries, not implementation details unnecessarily.**

A refactoring that preserves behavior should not require rewriting large numbers of tests simply because internal implementation details changed.

---

# 9. Git & GitHub Strategy

## 9.1 Objective

Git and GitHub will be used from the beginning as part of the engineering workflow, not merely as a final code-upload mechanism.

The repository history should demonstrate progressive engineering work.

---

## 9.2 Repository Strategy

Repository:

`rag-performance-optimization-system`

The repository should contain the complete project source, tests, configuration, and documentation required to reproduce the system.

Generated or unnecessary artifacts should not be committed.

Secrets and credentials must never be committed.

---

## 9.3 Development Workflow

The general workflow will be:

```text
Understand
    ↓
Design
    ↓
Implement Small Change
    ↓
Run
    ↓
Test
    ↓
Review
    ↓
Quality Checks
    ↓
Commit
    ↓
Push
```

This workflow is deliberately aligned with the project's learning methodology.

---

## 9.4 Commit Strategy

Commits should represent meaningful, logically coherent changes.

Examples:

* `chore: initialize project structure`
* `chore: configure development tooling`
* `feat: add document ingestion pipeline`
* `feat: add Chroma vector store integration`
* `feat: implement baseline RAG pipeline`
* `test: add retrieval integration tests`
* `refactor: separate retrieval application boundary`
* `feat: add request performance instrumentation`
* `feat: add response cache`
* `test: validate cache behavior`
* `docs: document caching experiment results`

Avoid vague commits such as:

* `stuff`
* `changes`
* `final`
* `fix`
* `update`
* `misc`

unless the commit message provides enough meaningful context.

---

## 9.5 Commit Granularity

Commits should generally be small enough that their purpose is obvious.

However, commits should not become artificially fragmented.

The goal is:

> **One coherent engineering change per commit.**

For example, implementing a feature and its directly related tests can reasonably belong to the same commit when they form one coherent change.

---

## 9.6 Git History as an Engineering Record

The Git history should tell a recognizable story:

```text
Foundation
    ↓
Baseline RAG
    ↓
Testing / Refactoring
    ↓
Observability
    ↓
Performance Baseline
    ↓
Optimization Experiments
    ↓
API
    ↓
UI
    ↓
Validation
    ↓
Documentation
```

This provides useful evidence of how the system evolved.

---

## 9.7 Branching Strategy

The project should avoid unnecessary Git complexity.

For a single-developer portfolio project, a simple workflow is appropriate.

The default approach can be:

```text
main
  │
  ├── focused development work
  │
  └── validated changes
```

Feature branches may be used when they provide a meaningful reason, such as:

* a larger isolated experiment
* an architectural refactor
* an optimization experiment requiring comparison
* a change that should be isolated before merging

A complex GitFlow-style workflow is unnecessary for this project.

---

## 9.8 Optimization Experiment History

Performance experiments should be traceable.

When an experiment becomes significant enough, Git should allow us to identify:

* the baseline implementation
* the experimental change
* the resulting measurements
* the final decision

This makes optimization work reproducible and defensible.

---

## 9.9 GitHub Documentation

The repository should eventually contain documentation explaining:

* what the project does
* how to run it
* architecture
* configuration
* baseline methodology
* optimization experiments
* results
* validation
* limitations

Documentation should evolve with the project rather than being written entirely at the end.

---

## 9.10 Git Hygiene

Before commits, verify:

* no secrets
* no unnecessary local files
* no virtual environments
* no generated caches
* no large temporary datasets
* no debugging artifacts
* no machine-specific configuration that should remain local

The repository should remain clean throughout development.

---

## 9.11 Git/GitHub Learning Objective

By the end of the project, the user should be comfortable with:

* repository initialization
* staging
* commits
* meaningful commit messages
* branches when justified
* merges when necessary
* remote repositories
* push/pull
* reviewing history
* reverting or correcting changes safely
* maintaining a professional project history

Git is therefore part of the engineering learning objective, not merely project administration.

# 10. Definition of Done

The project will be considered complete when the system, engineering practices, optimization experiments, validation, and documentation satisfy the following criteria.

Completion does not mean that every possible optimization has been implemented.

The project is complete when it demonstrates a **working, measurable, tested, optimized, and professionally engineered RAG system**, together with a clear understanding of the decisions behind it.

---

## 10.1 Functional Completion

The system must provide a working RAG pipeline capable of:

* ingesting the selected document corpus
* chunking documents
* generating embeddings
* storing embeddings in ChromaDB
* retrieving relevant information
* constructing context
* generating answers through Ollama
* returning useful retrieval information
* handling expected failure cases

The baseline RAG must be reproducible using the documented configuration.

---

## 10.2 Architecture Completion

The final architecture should demonstrate:

* clear separation of responsibilities
* modular Python components
* appropriate dependency boundaries
* reasonable dependency direction
* appropriate use of dependency inversion
* appropriate use of interfaces/protocols
* separation between application logic and infrastructure
* separation between API and application logic
* maintainable configuration

The architecture should remain understandable.

Complexity that cannot be justified should be removed.

---

## 10.3 Software Engineering Completion

The project should demonstrate practical application of:

* Clean Code
* SOLID principles
* separation of concerns
* high cohesion
* low coupling
* type hints
* dependency management
* configuration management
* error handling
* logging
* testability
* modular design

These principles should be applied pragmatically rather than mechanically.

---

## 10.4 Testing Completion

The project should have an appropriate automated testing foundation including:

* unit tests
* integration tests
* RAG workflow tests
* API tests
* regression validation

The exact number of tests is not a completion criterion.

The important requirement is that critical behavior and important boundaries have meaningful validation.

---

## 10.5 Quality Tooling Completion

The project should have functioning quality checks using the selected tooling:

* Ruff
* MyPy
* Pyright
* pytest
* pre-commit

The final codebase should pass the agreed project checks.

Any intentional exceptions should be documented and justified.

---

## 10.6 Observability Completion

The system should provide enough instrumentation to understand the major performance characteristics of a RAG request.

At minimum, the project should be able to measure relevant aspects of:

* retrieval latency
* context size
* token usage where available
* LLM latency
* TTFT where practical
* total request latency
* relevant errors
* configuration/model information

The goal is not exhaustive telemetry.

The goal is sufficient observability to support engineering decisions.

---

## 10.7 Baseline Completion

A documented baseline must exist before final optimization conclusions are made.

The baseline should identify:

* configuration
* workload/evaluation set
* relevant performance measurements
* retrieval behavior
* answer behavior
* observed bottlenecks
* important limitations

The baseline becomes the reference point for optimization experiments.

---

## 10.8 Optimization Completion

The project should contain a meaningful set of selected optimization experiments.

The exact number is intentionally not fixed.

At least some meaningful optimization should be investigated in both areas:

### LLM / Generation

Examples:

* token/context optimization
* caching
* streaming
* generation configuration
* model selection

### RAG / Retrieval

Examples:

* retrieval `k`
* chunking
* retrieval efficiency
* embedding efficiency
* vector-search behavior

Only experiments justified by the baseline need to be implemented.

---

## 10.9 Experiment Completion

Each significant optimization experiment should document:

* hypothesis
* baseline
* change
* measurements
* quality validation
* result
* decision
* trade-offs

Experiments may conclude:

* accepted
* rejected
* inconclusive
* neutral

Rejected experiments are still valuable evidence of engineering reasoning.

---

## 10.10 Quality and Regression Completion

The final optimized system should be evaluated against the golden dataset.

The evaluation should demonstrate that selected optimizations do not introduce unacceptable regressions in:

* retrieval behavior
* grounding
* expected answer behavior
* out-of-corpus behavior

Performance improvements must be interpreted together with useful behavior.

---

## 10.11 API Completion

FastAPI should provide a clean interface to the core system.

The API should demonstrate:

* request validation
* response schemas
* predictable errors
* health/status capability
* access to core RAG functionality
* automated API tests
* appropriate separation from the underlying application logic

---

## 10.12 UI Completion

The React interface should provide a small but functional interface for:

* asking questions
* displaying answers
* inspecting retrieved context
* viewing relevant performance information
* viewing selected experiment information

The UI should remain intentionally lightweight.

---

## 10.13 Documentation Completion

The repository documentation should explain:

* project objective
* architecture
* technology stack
* setup
* configuration
* usage
* baseline RAG
* measurement methodology
* baseline results
* optimization experiments
* optimization results
* trade-offs
* validation methodology
* limitations
* future work

A technically competent developer should be able to understand the project without access to the original development conversation.

---

## 10.14 Git/GitHub Completion

The final repository should demonstrate:

* meaningful commits
* clean history
* appropriate Git hygiene
* no secrets
* no unnecessary artifacts
* reproducible project setup
* complete source and documentation

The Git history should reflect the progressive development of the system.

---

## 10.15 Learning Completion

This is one of the most important completion criteria.

The user should be able to explain:

### RAG

* how the complete RAG pipeline works
* why chunking exists
* why embeddings are needed
* how vector retrieval works
* how context reaches the LLM
* what LangChain is doing

### Python

* the important modules
* major classes/functions
* important types
* dependency relationships
* error-handling decisions

### Software Engineering

* why the architecture is structured as it is
* where SOLID principles are applied
* where they were intentionally not applied
* why abstractions exist
* why unnecessary abstractions were rejected

### Performance

* what the baseline bottlenecks were
* how they were identified
* why specific optimizations were selected
* how experiments were designed
* what changed
* what trade-offs resulted

### Testing

* what the tests validate
* why unit versus integration tests exist
* how regression is evaluated
* how optimization quality is validated

### Engineering Decisions

The user should be able to defend important architectural and optimization decisions in a technical interview.

---

# 11. Project Risks & Scope-Control Rules

## 11.1 Primary Risk: Overengineering

The largest project risk is unnecessary complexity.

Because the project combines:

* RAG
* LangChain
* software engineering
* optimization
* observability
* FastAPI
* React
* testing

there is a natural tendency to keep adding abstractions and technologies.

The project must actively resist this.

### Rule

> **Do not add complexity merely because it is technically possible.**

---

## 11.2 Scope-Creep Rule

A proposed feature must satisfy at least one of these conditions:

1. It directly supports a project objective.
2. It provides significant learning value.
3. It is necessary to validate an important hypothesis.
4. It solves a concrete problem discovered during implementation.

If none applies, it should probably remain out of scope.

---

## 11.3 Baseline-First Rule

No significant optimization should be introduced simply because it is considered a best practice.

The sequence is:

```text id="e2r0uo"
Baseline
   ↓
Measure
   ↓
Identify Problem
   ↓
Form Hypothesis
   ↓
Optimize
   ↓
Measure Again
   ↓
Validate Quality
```

This is a central rule of the project.

---

## 11.4 No Premature Abstraction

Do not create abstractions solely because:

* SOLID recommends interfaces
* enterprise architecture commonly uses factories
* dependency injection frameworks exist
* a future implementation might someday be different

An abstraction should have a concrete justification.

Examples of valid reasons may include:

* replacing an external dependency
* improving testability
* isolating infrastructure
* supporting a real experiment
* separating responsibilities that have become difficult to maintain

---

## 11.5 No Technology Collection

The project should not become a demonstration of how many AI technologies can be combined.

The following should not be added merely for portfolio keywords:

* LangGraph
* agent frameworks
* GraphRAG
* hybrid search
* rerankers
* multiple vector databases
* multiple observability platforms
* complex evaluation frameworks
* cloud infrastructure
* Kubernetes
* unnecessary databases

A technology belongs in the project only when its role is justified.

---

## 11.6 LangChain Scope Rule

LangChain should be used meaningfully, but it should not become an abstraction layer that hides the entire system from the learner.

For important components, the user should understand:

* what LangChain provides
* what happens underneath
* where LangChain helps
* where direct implementation may be clearer

The project should avoid blindly wrapping every operation in LangChain abstractions.

---

## 11.7 Performance Measurement Rule

Performance numbers should always be interpreted in context.

Local measurements depend on:

* CPU
* available RAM
* GPU availability
* operating system
* model
* model configuration
* document corpus
* workload
* concurrent activity

Therefore:

> **Performance results in this project are experimental results for the defined environment, not universal benchmarks.**

---

## 11.8 Quality-vs-Performance Rule

A faster system is not automatically a better system.

Optimization decisions should consider:

```text id="vqqkqv"
Latency
   +
Resource Usage
   +
Retrieval Quality
   +
Answer Quality
   +
System Complexity
```

An optimization that produces a small performance improvement while significantly increasing complexity may be rejected.

---

## 11.9 Observability Scope Rule

Do not measure everything simply because measurement is possible.

Instrumentation should answer useful engineering questions.

For every metric, ask:

> What decision will this metric help us make?

If there is no meaningful answer, the metric may not belong in the system.

---

## 11.10 React Scope Rule

The frontend should remain small.

If frontend work starts consuming disproportionate project time, the scope should be reduced rather than allowing the UI to become a second major project.

The primary learning value remains:

* RAG engineering
* Python engineering
* performance engineering
* optimization
* testing

---

## 11.11 Experiment Scope Rule

Candidate experiments are not commitments.

The project may identify many possible optimizations but implement only the ones supported by evidence.

The decision process is:

```text id="ubt6sm"
Candidate
    ↓
Baseline Evidence
    ↓
Learning Value
    ↓
Expected Impact
    ↓
Implementation Cost
    ↓
Select / Reject
```

---

## 11.12 Reproducibility Rule

Performance experiments should be conducted under controlled and documented conditions whenever practical.

Record relevant:

* model
* model configuration
* retrieval configuration
* chunking configuration
* corpus
* hardware/environment
* workload
* optimization change

This makes comparisons more meaningful.

---

## 11.13 Learning-Ownership Rule

The assistant should support the implementation but should not replace the user's understanding.

The preferred workflow is:

```text id="x2e4sl"
Understand
    ↓
Design Together
    ↓
Small Guidance
    ↓
User Implements
    ↓
Run
    ↓
Test
    ↓
Review
    ↓
Refine
```

Large sections of opaque code should be avoided.

The user should own the final implementation and understand the important parts of it.

---

## 11.14 No Optimization Before Understanding

A recurring project principle is:

> **Do not optimize what you do not understand.**

Before optimizing a component, the user should understand:

* what the component does
* how it interacts with the rest of the system
* what metric represents its behavior
* what trade-offs an optimization may introduce

---

## 11.15 No Scope Expansion During Implementation Without Review

If implementation reveals a potentially valuable new direction, it should first be classified as:

* required
* useful but optional
* future work
* out of scope

It should not automatically become part of the current phase.

This prevents the project from continuously expanding.

---

## 11.16 Final Scope-Control Principle

The project should optimize for:

> **Depth of understanding and engineering quality, not breadth of technologies.**

A smaller system that the user can explain, measure, test, optimize, and defend technically is more valuable than a much larger system whose architecture and behavior are only partially understood.
