# Module: prompt_builder
# Path: src/rag_performance/prompt_builder.py
# Purpose: Construct prompts for retrieval-augmented generation.
# Description: Build context-grounded prompts that encourage direct,
# complete, relevant, and appropriately structured answers.


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
            "- Answer the exact question directly before providing "
            "additional details.\n"
            "- Cover all material requirements needed to answer the "
            "question; do not stop after mentioning only one requirement.\n"
            "- For questions involving multiple requirements, steps, "
            "or conditions, use a numbered list or checklist.\n"
            "- Include only details relevant to the question; avoid "
            "unnecessary information from other parts of the context.\n"
            "- Distinguish mandatory rules from recommendations and "
            "conditional requirements.\n"
            "- Base your answer only on the provided context.\n"
            "- Do not invent facts or make unsupported assumptions.\n"
            "- If the context does not contain enough information "
            "to answer the question, say so clearly.\n"
            "- Identify relevant sources when source information "
            "is provided.\n\n"
            f"Context:\n{context.strip()}\n\n"
            f"Question:\n{query.strip()}\n\n"
            "Answer:"
        )
