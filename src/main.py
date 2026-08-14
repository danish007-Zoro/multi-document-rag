import os

from rag_pipeline import RAGPipeline


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)

DOCUMENTS = [
    os.path.join(
        DATA_DIR,
        "Sample1.pdf"
    ),
    os.path.join(
        DATA_DIR,
        "Sample2.pdf"
    ),
    os.path.join(
        DATA_DIR,
        "Sample3.pdf"
    )
]

DISTANCE_THRESHOLD = 1.4
TOP_K = 3


# ==================================================
# VALIDATE DOCUMENTS
# ==================================================

def validate_documents():

    for path in DOCUMENTS:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Document not found: {path}"
            )


# ==================================================
# DISPLAY STATISTICS
# ==================================================

def display_statistics(stats):

    print()
    print("=" * 70)
    print("RAG PIPELINE READY")
    print("=" * 70)
    print()

    print(
        f"Documents : "
        f"{stats['total_documents']}"
    )

    print(
        f"Pages     : "
        f"{stats['total_pages']}"
    )

    print(
        f"Chunks    : "
        f"{stats['total_chunks']}"
    )

    print(
        f"Embedding : "
        f"{stats['embedding_dimension']}"
    )

    print(
        f"FAISS     : "
        f"{stats['vectors']}"
    )

    print(
        f"Top-K     : "
        f"{TOP_K}"
    )

    print(
        f"Threshold : "
        f"{DISTANCE_THRESHOLD}"
    )


# ==================================================
# DISPLAY SOURCES
# ==================================================

def display_sources(sources):

    if not sources:

        print(
            "Sources: None"
        )

        return

    print()
    print("Sources:")
    print()

    for index, source in enumerate(
        sources,
        start=1
    ):

        print(
            f"[{index}] "
            f"{source['document']} | "
            f"Page {source['page']} | "
            f"Chunk {source['chunk_id']} | "
            f"Distance {source['distance']:.4f}"
        )


# ==================================================
# ASK QUESTION
# ==================================================

def ask_question(
    pipeline,
    query
):

    result = pipeline.ask(
        query
    )

    print()
    print("-" * 70)

    print(
        f"Question: {query}"
    )

    print()

    print(
        f"Accepted: "
        f"{result['accepted']}"
    )

    print(
        f"Best Distance: "
        f"{result['best_distance']:.4f}"
    )

    print()

    print("Answer:")
    print(
        result["answer"]
    )

    display_sources(
        result["sources"]
    )

    print("-" * 70)


# ==================================================
# DOCUMENT SUMMARY
# ==================================================

def summarize_document(
    pipeline,
    document_name
):

    summary = pipeline.summarize(
        document_name
    )

    print()
    print("=" * 70)
    print(
        f"SUMMARY: {document_name}"
    )
    print("=" * 70)
    print()

    print(summary)


# ==================================================
# INTERACTIVE APPLICATION
# ==================================================

def interactive_mode(
    pipeline
):

    print()
    print("=" * 70)
    print("MULTI-DOCUMENT RAG SYSTEM")
    print("=" * 70)

    print()
    print("Commands:")
    print("  ask <question>")
    print("  summarize <document>")
    print("  documents")
    print("  stats")
    print("  exit")

    while True:

        print()

        user_input = input(
            "RAG> "
        ).strip()

        if not user_input:

            continue

        if user_input.lower() == "exit":

            print(
                "Exiting RAG system."
            )

            break


        if user_input.lower() == "documents":

            print()
            print("Loaded documents:")

            for document in pipeline.documents:

                print(
                    f"  - {document['name']}"
                )

            continue


        if user_input.lower() == "stats":

            display_statistics(
                {
                    "total_documents":
                        len(pipeline.documents),

                    "total_pages":
                        pipeline.total_pages,

                    "total_chunks":
                        len(pipeline.chunks),

                    "embedding_dimension":
                        (
                            pipeline.index.d
                            if pipeline.index is not None
                            else "N/A"
                        ),

                    "vectors":
                        (
                            pipeline.index.ntotal
                            if pipeline.index is not None
                            else "N/A"
                        )
                }
            )

            continue


        if user_input.lower().startswith(
            "ask "
        ):

            query = user_input[
                4:
            ].strip()

            if not query:

                print(
                    "Please provide a question."
                )

                continue

            ask_question(
                pipeline,
                query
            )

            continue


        if user_input.lower().startswith(
            "summarize "
        ):

            document_name = user_input[
                10:
            ].strip()

            if not document_name:

                print(
                    "Please provide a document name."
                )

                continue

            try:

                summarize_document(
                    pipeline,
                    document_name
                )

            except ValueError as error:

                print(
                    f"Error: {error}"
                )

            continue


        print(
            "Unknown command."
        )

        print(
            "Use: ask, summarize, "
            "documents, stats, or exit."
        )


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)
    print("STARTING MULTI-DOCUMENT RAG SYSTEM")
    print("=" * 70)

    print()
    print("Validating documents...")

    validate_documents()

    for path in DOCUMENTS:

        print(
            f"✓ {os.path.basename(path)}"
        )


    # ----------------------------------------------
    # Build pipeline
    # ----------------------------------------------

    print()
    print("Loading documents...")
    print()

    pipeline = RAGPipeline(
        distance_threshold=DISTANCE_THRESHOLD,
        top_k=TOP_K
    )

    stats = pipeline.load_documents(
        DOCUMENTS
    )


    # ----------------------------------------------
    # Display pipeline statistics
    # ----------------------------------------------

    display_statistics(
        stats
    )


    # ----------------------------------------------
    # Start interactive application
    # ----------------------------------------------

    interactive_mode(
        pipeline
    )


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    main()