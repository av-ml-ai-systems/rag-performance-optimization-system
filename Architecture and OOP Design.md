# Architecture and OOP Design

## Purpose

Define and approve the software architecture and Object-Oriented Programming (OOP) design for Phase 2 of the RAG Performance Optimization System before implementation begins.

## Design Scope

The document will define:

1. **End-to-end architecture:** How documents move through ingestion, chunking, embedding generation, vector storage, retrieval, context construction, and answer generation with Ollama.
2. **Class responsibilities:** Which classes are needed and the responsibilities of each.
3. **Class relationships:** How components collaborate, including composition and interfaces where justified.
4. **Module boundaries:** How the code is organized into modules and packages.
5. **Dependency management:** How dependencies flow between components and how external systems are integrated.
6. **Testing strategy:** The unit, integration, and functional tests required to validate the design and implementation.

## Design Principles

- Introduce OOP from the beginning rather than adding it after implementation.
- Prefer cohesive classes with clear, well-defined responsibilities.
- Use composition and dependency injection when they provide a practical benefit.
- Avoid unnecessary abstractions, interfaces, and classes.
- Keep the architecture understandable, maintainable, and testable.
- Use LangChain where it provides meaningful functionality rather than introducing it unnecessarily.

# Architecture and OOP Design

## 1. End-to-End Architecture

The system follows a Retrieval-Augmented Generation (RAG) architecture that retrieves relevant information from documents and uses it to generate context-aware answers.

### Document Ingestion Pipeline

1. **Document ingestion:** Load source documents.
2. **Text chunking:** Split documents into smaller text segments while preserving relevant metadata.
3. **Embedding generation:** Convert each chunk into a numerical vector using an embedding model.
4. **Vector storage:** Store vectors, text chunks, and metadata in ChromaDB.

### Query Processing Pipeline

5. **Query input:** Receive a user's question.
6. **Query embedding:** Convert the question into a vector using the same embedding model used for document chunks.
7. **Retrieval:** Search ChromaDB for the most relevant document chunks.
8. **Context construction:** Assemble the retrieved chunks into a context for the language model.
9. **Prompt construction:** Combine the question, retrieved context, and instructions into a prompt.
10. **Answer generation:** Send the prompt to an Ollama-hosted language model.
11. **Response delivery:** Return the generated answer to the user.

Document ingestion and query processing are separate workflows. Documents are ingested into the vector store before questions are answered; the system does not need to reprocess all documents for every query.

## 2. OOP Design and Class Responsibilities

The system will use cohesive classes with clearly defined responsibilities.

- **DocumentLoader:** Loads source documents.
- **TextChunker:** Splits document text into chunks and preserves metadata.
- **EmbeddingProvider:** Generates embeddings for document chunks and user queries.
- **VectorStore:** Stores document chunks and embeddings and retrieves relevant results through ChromaDB.
- **Retriever:** Coordinates query embedding and relevant document retrieval.
- **ContextBuilder:** Builds context from retrieved chunks.
- **PromptBuilder:** Constructs the prompt supplied to the language model.
- **AnswerGenerator:** Interacts with Ollama to generate answers.
- **RAGPipeline:** Coordinates the query-processing workflow and connects the components.

These are proposed responsibilities, not a requirement to create a separate class for every operation. We will refine the class structure during design and avoid abstractions that do not provide a clear benefit.

## 3. Module Boundaries and Dependencies

Organize the implementation under `src/rag_performance/` by responsibility.

- **Ingestion:** Document loading and chunking.
- **Embeddings:** Embedding generation.
- **Storage and retrieval:** ChromaDB integration and relevant document retrieval.
- **Generation:** Context building, prompt construction, and Ollama integration.
- **Pipeline:** Coordination of the end-to-end workflow.

The final module structure will follow the approved class design. Avoid circular dependencies and keep external integrations isolated behind clear component boundaries.

## 4. External Components and Technology Stack

- **Python:** Core implementation language.
- **LangChain:** Used selectively for useful integrations and abstractions.
- **ChromaDB:** Vector storage and similarity search.
- **Ollama:** Local language model inference.
- **Embedding model:** Converts document chunks and queries into vectors.
- **pytest:** Unit, integration, and functional testing.
- **Ruff, MyPy, and Pyright:** Code quality, type checking, and static analysis.

The embedding model, language model, and relevant configuration will be selected and documented before implementation. The embedding model must remain consistent between document ingestion and query processing.

## 5. Testing Strategy

Testing will be introduced alongside implementation.

- **Unit tests:** Validate individual components in isolation.
- **Integration tests:** Validate interactions with ChromaDB, the embedding model, and Ollama.
- **Functional tests:** Validate document ingestion, retrieval, and answer generation.
- **End-to-end validation:** Confirm that a user question produces an answer grounded in retrieved document content.

External dependencies should be replaceable with test doubles where practical, without introducing unnecessary abstractions.

## 6. Design Principles

- Apply OOP from the beginning.
- Give each class a clear and cohesive responsibility.
- Prefer composition over unnecessary inheritance.
- Use dependency injection where it improves testability or flexibility.
- Keep external integrations separate from core workflow coordination.
- Use LangChain when it provides practical value.
- Avoid overengineering and premature optimization.
- Keep the implementation understandable, maintainable, and testable.