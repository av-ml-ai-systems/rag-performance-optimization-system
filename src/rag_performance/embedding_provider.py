# Module: embedding_provider
# Path: src/rag_performance/embedding_provider.py
# Purpose: Define the contract for interchangeable embedding providers.
# Description: Extend LangChain's embedding interface so implementations
# can be injected directly into vector stores and other RAG components.

from abc import ABC

from langchain_core.embeddings import Embeddings


class EmbeddingProvider(Embeddings, ABC):
    """Abstract contract for interchangeable embedding providers."""
