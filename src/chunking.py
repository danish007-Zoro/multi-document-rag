import re


def clean_page_text(text):
    """
    Clean extracted PDF text while preserving
    meaningful paragraph structure.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Remove excessive spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def is_reference_section(text):
    """
    Detect whether a page section is primarily
    a references/bibliography section.

    Returns:
        bool
    """

    stripped = text.strip()

    if not stripped:
        return False

    first_lines = "\n".join(
        stripped.splitlines()[:5]
    ).lower()

    reference_markers = [
        "references:",
        "references",
        "bibliography",
        "works cited"
    ]

    for marker in reference_markers:

        if marker in first_lines:

            return True

    # Pages containing many URLs are likely
    # reference pages.

    url_count = len(
        re.findall(
            r"https?://",
            stripped
        )
    )

    if url_count >= 3:

        return True

    return False


def split_into_sentences(text):
    """
    Split text into approximately sentence-level units.

    This is intentionally simple and designed for
    normal PDF prose rather than perfect NLP parsing.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    pages,
    chunk_size=500,
    overlap=100
):
    """
    Split document pages into retrieval-friendly chunks.

    Improvements over the previous implementation:

    1. Cleans extracted PDF text.
    2. Detects and skips reference-only pages.
    3. Uses sentence-aware chunk boundaries.
    4. Preserves page metadata.
    5. Maintains approximate chunk size.
    6. Adds overlap between chunks.

    Args:
        pages:
            List containing page text.

        chunk_size:
            Approximate maximum character size.

        overlap:
            Approximate amount of overlapping text.

    Returns:
        list:
            Chunk dictionaries containing metadata.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )


    chunks = []

    chunk_id = 1


    # ==================================================
    # PROCESS EACH PAGE
    # ==================================================

    for page_number, raw_page in enumerate(
        pages,
        start=1
    ):

        page = clean_page_text(
            raw_page
        )

        if not page:
            continue


        # ----------------------------------------------
        # Skip reference-only pages
        # ----------------------------------------------

        if is_reference_section(
            page
        ):
            continue


        # ----------------------------------------------
        # Split page into sentences
        # ----------------------------------------------

        sentences = split_into_sentences(
            page
        )

        if not sentences:
            continue


        # ----------------------------------------------
        # Build chunks from sentences
        # ----------------------------------------------

        current_sentences = []
        current_length = 0


        for sentence in sentences:

            sentence_length = len(
                sentence
            )

            # ------------------------------------------
            # Handle very large individual sentences
            # ------------------------------------------

            if sentence_length > chunk_size:

                if current_sentences:

                    chunk_text_value = " ".join(
                        current_sentences
                    )

                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "page": page_number,
                            "text": chunk_text_value
                        }
                    )

                    chunk_id += 1

                    current_sentences = []
                    current_length = 0


                # Split oversized sentence safely
                for start in range(
                    0,
                    sentence_length,
                    chunk_size - overlap
                ):

                    piece = sentence[
                        start:
                        start + chunk_size
                    ].strip()

                    if not piece:
                        continue

                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "page": page_number,
                            "text": piece
                        }
                    )

                    chunk_id += 1

                continue


            # ------------------------------------------
            # Check whether sentence fits
            # ------------------------------------------

            proposed_length = (
                current_length
                +
                sentence_length
                +
                (
                    1
                    if current_sentences
                    else 0
                )
            )


            if (
                current_sentences
                and
                proposed_length > chunk_size
            ):

                # --------------------------------------
                # Store current chunk
                # --------------------------------------

                chunk_text_value = " ".join(
                    current_sentences
                )

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "text": chunk_text_value
                    }
                )

                chunk_id += 1


                # --------------------------------------
                # Build overlap from previous sentences
                # --------------------------------------

                overlap_sentences = []

                overlap_length = 0


                for previous_sentence in reversed(
                    current_sentences
                ):

                    if (
                        overlap_length
                        +
                        len(previous_sentence)
                        >
                        overlap
                    ):

                        break

                    overlap_sentences.insert(
                        0,
                        previous_sentence
                    )

                    overlap_length += (
                        len(previous_sentence)
                        + 1
                    )


                current_sentences = (
                    overlap_sentences
                )

                current_length = (
                    overlap_length
                )


            # ------------------------------------------
            # Add new sentence
            # ------------------------------------------

            current_sentences.append(
                sentence
            )

            current_length += (
                sentence_length
                +
                (
                    1
                    if len(current_sentences) > 1
                    else 0
                )
            )


        # ----------------------------------------------
        # Store remaining sentences
        # ----------------------------------------------

        if current_sentences:

            chunk_text_value = " ".join(
                current_sentences
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": chunk_text_value
                }
            )

            chunk_id += 1


    return chunks