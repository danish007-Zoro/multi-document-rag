import os
import sys


# ==================================================
# PATH SETUP
# ==================================================

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SRC_DIR
)

if SRC_DIR not in sys.path:

    sys.path.insert(
        0,
        SRC_DIR
    )


# ==================================================
# IMPORTS
# ==================================================

from rag_pipeline import (
    RAGPipeline
)

from groundedness import (
    evaluate_groundedness
)


# ==================================================
# CONFIGURATION
# ==================================================

PDF_PATHS = [

    os.path.join(
        PROJECT_ROOT,
        "data",
        "Sample1.pdf"
    ),

    os.path.join(
        PROJECT_ROOT,
        "data",
        "Sample2.pdf"
    ),

    os.path.join(
        PROJECT_ROOT,
        "data",
        "Sample3.pdf"
    )
]


TOP_K = 3

DISTANCE_THRESHOLD = 1.4


TEST_QUERIES = [

    "What is Artificial Intelligence?",

    "What is Machine Learning?",

    "When was the Turing test invented?",

    "When was Siri announced?",

    "When was OpenAI founded?",

    "What are the applications of machine learning?",

    "How is machine learning used in gaming?",

    "How does machine learning work?"
]


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)

    print(
        "RAG GROUNDEDNESS EVALUATION"
    )

    print("=" * 70)


    # ------------------------------------------------
    # Validate documents
    # ------------------------------------------------

    print("\nValidating documents...")


    for path in PDF_PATHS:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Document not found: {path}"
            )


        print(
            f"✓ {os.path.basename(path)}"
        )


    # ------------------------------------------------
    # Load documents
    # ------------------------------------------------

    print("\nLoading documents...")


    pipeline = RAGPipeline(
        distance_threshold=
            DISTANCE_THRESHOLD,

        top_k=
            TOP_K
    )


    stats = pipeline.load_documents(
        PDF_PATHS
    )


    # ==================================================
    # DOCUMENT STATISTICS
    # ==================================================

    print("\n")

    print("=" * 70)

    print(
        "DOCUMENT STATISTICS"
    )

    print("=" * 70)


    print(
        f"\nDocuments : "
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
        f"Retrieval Threshold : "
        f"{DISTANCE_THRESHOLD}"
    )

    print(
        "Groundedness Method : "
        "NLI Entailment"
    )


    # ==================================================
    # EVALUATION
    # ==================================================

    total_claims = 0

    grounded_claims = 0

    ungrounded_claims = 0


    for number, query in enumerate(
        TEST_QUERIES,
        start=1
    ):

        print("\n")

        print("=" * 70)

        print(
            f"[{number:02d}] {query}"
        )

        print("=" * 70)


        # --------------------------------------------
        # Ask RAG
        # --------------------------------------------

        result = pipeline.ask(
            query
        )


        print(
            f"\nAccepted      : "
            f"{result['accepted']}"
        )

        print(
            f"Best Distance : "
            f"{result['best_distance']:.4f}"
        )


        print(
            "\nAnswer:"
        )

        print(
            result["answer"]
        )


        # --------------------------------------------
        # Groundedness
        # --------------------------------------------

        groundedness = (
            evaluate_groundedness(
                result["answer"],
                result["results"],
                pipeline.model,
                0.55
            )
        )


        total = (
            groundedness[
                "total_sentences"
            ]
        )

        grounded = (
            groundedness[
                "grounded_sentences"
            ]
        )

        ungrounded = (
            groundedness[
                "ungrounded_sentences"
            ]
        )

        score = (
            groundedness[
                "groundedness_score"
            ]
        )


        total_claims += total

        grounded_claims += grounded

        ungrounded_claims += ungrounded


        # --------------------------------------------
        # Groundedness summary
        # --------------------------------------------

        print(
            "\nGroundedness:"
        )

        print(
            f"Total Claims        : "
            f"{total}"
        )

        print(
            f"Grounded Claims     : "
            f"{grounded}"
        )

        print(
            f"Ungrounded Claims   : "
            f"{ungrounded}"
        )

        print(
            f"Groundedness Score  : "
            f"{score * 100:.2f}%"
        )


        # --------------------------------------------
        # Claim analysis
        # --------------------------------------------

        print(
            "\nClaim Analysis:"
        )


        for item in groundedness[
            "sentence_results"
        ]:

            if item["grounded"]:

                status = "GROUNDED"

            else:

                status = "UNGROUNDED"


            print(
                f"\n[{status}] "
                f"NLI Label: "
                f"{item['nli_label']}"
            )


            print(
                f"Confidence: "
                f"{item['nli_confidence']:.4f}"
            )


            print(
                f"Claim:\n"
                f"  {item['sentence']}"
            )


            # ----------------------------------------
            # Evidence provenance
            # ----------------------------------------

            evidence = (
                item["evidence"]
            )


            if evidence:

                print(
                    "\nEvidence:"
                )

                print(
                    f"  Document : "
                    f"{evidence['document']}"
                )

                print(
                    f"  Page     : "
                    f"{evidence['page']}"
                )

                print(
                    f"  Chunk    : "
                    f"{evidence['chunk_id']}"
                )


    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print("\n")

    print("=" * 70)

    print(
        "GROUNDEDNESS EVALUATION SUMMARY"
    )

    print("=" * 70)


    print(
        f"\nTotal Claims      : "
        f"{total_claims}"
    )

    print(
        f"Grounded Claims   : "
        f"{grounded_claims}"
    )

    print(
        f"Ungrounded Claims : "
        f"{ungrounded_claims}"
    )


    if total_claims > 0:

        overall_score = (
            grounded_claims
            /
            total_claims
        )

    else:

        overall_score = 0.0


    print(
        f"Overall Groundedness : "
        f"{overall_score * 100:.2f}%"
    )


    print("\n")

    print("=" * 70)

    print(
        "GROUNDEDNESS EVALUATION COMPLETE"
    )

    print("=" * 70)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    main()