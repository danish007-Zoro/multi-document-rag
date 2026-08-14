from generation import client


MODEL_NAME = "llama-3.1-8b-instant"


def _generate_summary(prompt, max_tokens=350):
    """
    Generate a summary using the Groq LLM.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()


def _split_pages(pages, max_chars=7000):
    """
    Group document pages into manageable sections.

    The goal is to prevent the summarization request
    from exceeding the LLM context/rate limits.
    """

    sections = []
    current_section = []
    current_chars = 0

    for page_number, page_text in enumerate(
        pages,
        start=1
    ):

        if not page_text or not page_text.strip():
            continue

        page_text = page_text.strip()

        page_block = (
            f"[Page {page_number}]\n"
            f"{page_text}\n"
        )

        page_chars = len(page_block)

        if (
            current_section
            and current_chars + page_chars > max_chars
        ):
            sections.append(
                "\n".join(current_section)
            )

            current_section = []
            current_chars = 0

        current_section.append(page_block)
        current_chars += page_chars

    if current_section:
        sections.append(
            "\n".join(current_section)
        )

    return sections


def summarize_document(
    document_name,
    pages
):
    """
    Generate a grounded summary of an entire document.

    Uses hierarchical summarization to avoid sending
    the entire document in a single oversized LLM request.
    """

    if not pages:
        raise ValueError(
            "Document contains no extractable text."
        )

    # --------------------------------------------------
    # 1. Split document into manageable sections
    # --------------------------------------------------

    sections = _split_pages(
        pages,
        max_chars=7000
    )

    if not sections:
        raise ValueError(
            "Document contains no extractable text."
        )

    # --------------------------------------------------
    # 2. Summarize each section
    # --------------------------------------------------

    section_summaries = []

    for index, section in enumerate(
        sections,
        start=1
    ):

        prompt = f"""
You are a document summarization assistant.

Summarize ONLY the information contained in the
provided section of the document.

Document: {document_name}

SECTION {index}

{section}

Instructions:

1. Identify the most important concepts and ideas.
2. Preserve important facts, names, dates and examples.
3. Do not introduce information that is not present.
4. Keep the summary concise.
5. Focus on information useful for understanding
   the document as a whole.

Return a concise section summary.
"""

        summary = _generate_summary(
            prompt,
            max_tokens=350
        )

        section_summaries.append(
            summary
        )

    # --------------------------------------------------
    # 3. Combine section summaries
    # --------------------------------------------------

    combined_summaries = "\n\n".join(
        f"[Section {index}]\n{summary}"
        for index, summary in enumerate(
            section_summaries,
            start=1
        )
    )

    # --------------------------------------------------
    # 4. Generate final document summary
    # --------------------------------------------------

    final_prompt = f"""
You are a document summarization assistant.

Create a final grounded summary of the document
using ONLY the section summaries provided below.

Document: {document_name}

SECTION SUMMARIES:

{combined_summaries}

Instructions:

1. Give a concise overview of the document.
2. Identify the most important concepts and ideas.
3. Preserve important facts, names, dates and examples.
4. Do not introduce information that is not present
   in the section summaries.
5. If the document contains multiple major topics,
   organize them clearly.
6. Keep the summary easy to scan.

Use this structure:

OVERVIEW

2-4 sentences describing the document.


KEY POINTS

- Important point
- Important point
- Important point


IMPORTANT DETAILS

- Important fact, date, example or definition
- Important fact, date, example or definition

Keep the response concise.
"""

    return _generate_summary(
        final_prompt,
        max_tokens=500
    )