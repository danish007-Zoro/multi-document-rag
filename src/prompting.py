def build_context(results):
    """
    Build a structured context string from retrieved
    document chunks.

    Args:
        results (list):
            Retrieved chunks from FAISS.

    Returns:
        str:
            Formatted context for the LLM.
    """

    context_parts = []

    for chunk in results:

        document = chunk.get(
            "document",
            "Unknown Document"
        )

        page = chunk.get(
            "page",
            "Unknown"
        )

        chunk_id = chunk.get(
            "chunk_id",
            "Unknown"
        )

        context_parts.append(
            f"""
[Source {len(context_parts) + 1}]
Document: {document}
Page: {page}
Chunk: {chunk_id}

{chunk["text"]}
"""
        )

    return "\n".join(
        context_parts
    )


def build_prompt(query, context):
    """
    Build a grounded RAG prompt.

    The model is explicitly instructed to:
    - use only retrieved context
    - avoid outside knowledge
    - distinguish unsupported questions
    - synthesize multiple sources when necessary
    - avoid inventing citations or facts
    """

    prompt = f"""
You are a document-grounded question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the retrieved document
context.

STRICT RULES:

1. Do not use outside knowledge.

2. Do not make assumptions that are not supported
   by the retrieved context.

3. If the answer cannot be determined from the
   retrieved context, respond exactly with:

"I could not find the answer in the provided documents."

4. When multiple retrieved sources contain relevant
   information, combine them into one concise answer.

5. Do not mention information that is unrelated to
   the user's question.

6. Do not invent facts, dates, names, numbers,
   citations, or sources.

7. Keep the answer concise and factual.

8. Base every factual claim in your answer on the
   retrieved context.

RETRIEVED DOCUMENT CONTEXT
==========================

{context}

==========================

USER QUESTION
==========================

{query}

==========================

ANSWER
==========================
"""

    return prompt