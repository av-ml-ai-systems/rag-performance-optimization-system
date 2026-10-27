# Module: prompt_builder
# Path: src/rag_performance/prompt_builder.py
# Purpose: Construct prompts for retrieval-augmented generation.
# Description: Combine retrieved context and a user question into a consistent
# instruction that encourages grounded answers and transparent source usage.


class PromptBuilder:
    """Build prompts that ground answers in retrieved context."""

    def build(self, query: str, context: str) -> str:
        """Construct a prompt from a user query and context."""
        if not query.strip():
            raise ValueError("query must not be empty.")

        if not context.strip():
            raise ValueError("context must not be empty.")

        return (
            "You are a helpful assistant that answers questions using "
            "the provided context.\n"
            "Follow these instructions:\n"
            "- Base your answer on the provided context.\n"
            "- Do not invent facts that are not supported by the context.\n"
            "- If the context does not contain enough information, "
            "say so clearly.\n"
            "- Identify the relevant source when the context provides "
            "source information.\n\n"
            f"Context:\n{context.strip()}\n\n"
            f"Question:\n{query.strip()}\n\n"
            "Answer:"
        )
