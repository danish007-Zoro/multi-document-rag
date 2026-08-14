from generation import client


def summarize_document(
    document_name,
    pages
):
    """
    Generate a grounded summary of an entire document.

    Args:
        document_name (str):
            Name of the PDF document.

        pages (list):
            List containing the text of each page.

    Returns:
        str:
            Generated document summary.
    """

    if not pages:
        raise ValueError(
            "Document contains no extractable text."
        )

    # --------------------------------------------------
    # Build page-aware document context
    # --------------------------------------------------

    document_context = []

    for page_number, page_text in enumerate(
        pages,
        start=1
    ):

        if not page_text.strip():
            continue

        document_context.append(
            f"""
[Page {page_number}]
{page_text}
"""
        )

    context = "\n".join(
        document_context
    )

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = f"""
You are a document summarization assistant.

Summarize the provided document using ONLY the
information contained in the document.

Document: {document_name}

DOCUMENT CONTENT:
{context}

Instructions:

1. Give a concise overview of the document.
2. Identify the most important concepts and ideas.
3. Preserve important facts, names, dates and examples.
4. Do not introduce information that is not present
   in the document.
5. Do not mention that you are an AI.
6. Do not refer to the document as "the context".
7. If the document contains multiple major topics,
   organize them clearly.
8. Keep the summary easy to scan.

Use this structure:

OVERVIEW

[2-4 sentences]


KEY POINTS

• [important point]
• [important point]
• [important point]


IMPORTANT DETAILS

• [important fact, date, example or definition]
• [important fact, date, example or definition]

Keep the response concise while covering the
important information.
"""

    # --------------------------------------------------
    # Generate summary
    # --------------------------------------------------

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=700
    )

    return response.choices[0].message.content.strip()