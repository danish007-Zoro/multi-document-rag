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


# ==================================================
# ADVERSARIAL TEST CASES
# ==================================================

TEST_CASES = [

    # ------------------------------------------------
    # 01 - Correct claim
    # ------------------------------------------------

    {
        "query":
            "When was Siri announced?",

        "claim":
            "Siri was announced as a digital assistant "
            "by Apple in 2011.",

        "expected_grounded":
            True
    },


    # ------------------------------------------------
    # 02 - Wrong date
    # ------------------------------------------------

    {
        "query":
            "When was Siri announced?",

        "claim":
            "Siri was announced by Apple in 2008.",

        "expected_grounded":
            False
    },


    # ------------------------------------------------
    # 03 - Correct OpenAI claim
    # ------------------------------------------------

    {
        "query":
            "When was OpenAI founded?",

        "claim":
            "OpenAI was founded in 2015.",

        "expected_grounded":
            True
    },


    # ------------------------------------------------
    # 04 - Wrong person
    # ------------------------------------------------

    {
        "query":
            "When was OpenAI founded?",

        "claim":
            "OpenAI was founded by Bill Gates in 2015.",

        "expected_grounded":
            False
    },


    # ------------------------------------------------
    # 05 - Wrong date
    # ------------------------------------------------

    {
        "query":
            "When was OpenAI founded?",

        "claim":
            "OpenAI was founded in 2012.",

        "expected_grounded":
            False
    },


    # ------------------------------------------------
    # 06 - Correct Turing claim
    # ------------------------------------------------

    {
        "query":
            "When was the Turing test invented?",

        "claim":
            "The Turing test was invented in 1950 by Alan Turing.",

        "expected_grounded":
            True
    },


    # ------------------------------------------------
    # 07 - Wrong person
    # ------------------------------------------------

    {
        "query":
            "When was the Turing test invented?",

        "claim":
            "The Turing test was invented by Albert Einstein.",

        "expected_grounded":
            False
    },


    # ------------------------------------------------
    # 08 - Correct ML claim
    # ------------------------------------------------

    {
        "query":
            "What is Machine Learning?",

        "claim":
            "Machine learning gives computers the ability "
            "to learn without being explicitly programmed.",

        "expected_grounded":
            True
    },


    # ------------------------------------------------
    # 09 - Exaggeration
    # ------------------------------------------------

    {
        "query":
            "What is Machine Learning?",

        "claim":
            "Machine learning completely replaces "
            "human decision-making.",

        "expected_grounded":
            False
    },


    # ------------------------------------------------
    # 10 - Perfect accuracy claim
    # ------------------------------------------------

    {
        "query":
            "What is Machine Learning?",

        "claim":
            "Machine learning predicts outcomes with "
            "perfect accuracy.",

        "expected_grounded":
            False
    },


    # ------------------------------------------------
    # 11 - Exclusive claim
    # ------------------------------------------------

    {
        "query":
            "How is machine learning used in gaming?",

        "claim":
            "Machine learning is only used in gaming.",

        "expected_grounded":
            False
    },


    # ------------------------------------------------
    # 12 - Correct gaming claim
    # ------------------------------------------------

    {
        "query":
            "How is machine learning used in gaming?",

        "claim":
            "Machine learning is used in gaming to make "
            "games more interesting and create balanced "
            "gameplay.",

        "expected_grounded":
            True
    },


    # ------------------------------------------------
    # 13 - Correct application
    # ------------------------------------------------

    {
        "query":
            "What are the applications of machine learning?",

        "claim":
            "Machine learning can be used for fraud detection.",

        "expected_grounded":
            True
    },


    # ------------------------------------------------
    # 14 - Overgeneralization
    # ------------------------------------------------

    {
        "query":
            "What are the applications of machine learning?",

        "claim":
            "Machine learning is used exclusively for fraud detection.",

        "expected_grounded":
            False
    },


    # ------------------------------------------------
    # 15 - Unrelated claim
    # ------------------------------------------------

    {
        "query":
            "What is Artificial Intelligence?",

        "claim":
            "The capital of France is Paris.",

        "expected_grounded":
            False
    },


    # ------------------------------------------------
    # 16 - Wrong organization
    # ------------------------------------------------

    {
        "query":
            "When was Siri announced?",

        "claim":
            "Siri was created by Google.",

        "expected_grounded":
            False
    }

]


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)

    print(
        "ADVERSARIAL GROUNDEDNESS EVALUATION"
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
    # Load RAG pipeline
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


    # ==================================================
    # EVALUATION
    # ==================================================

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0


    print("\n")

    print("=" * 70)

    print(
        "ADVERSARIAL TEST CASES"
    )

    print("=" * 70)


    for number, case in enumerate(
        TEST_CASES,
        start=1
    ):

        print("\n")

        print(
            "-" * 70
        )

        print(
            f"[{number:02d}] "
            f"Query: {case['query']}"
        )

        print(
            f"Claim: "
            f"{case['claim']}"
        )


        # --------------------------------------------
        # Retrieve evidence
        # --------------------------------------------

        retrieval = pipeline.ask(
            case["query"]
        )


        print(
            f"\nRetrieval Accepted: "
            f"{retrieval['accepted']}"
        )

        print(
            f"Best Distance     : "
            f"{retrieval['best_distance']:.4f}"
        )


        # --------------------------------------------
        # Evaluate claim directly against retrieved
        # evidence
        # --------------------------------------------

        groundedness = evaluate_groundedness(
            case["claim"],
            retrieval["results"],
            pipeline.model,
            0.55
        )


        claim_result = None


        if groundedness[
            "sentence_results"
        ]:

            claim_result = groundedness[
                "sentence_results"
            ][0]


        if claim_result is None:

            predicted_grounded = False

            nli_label = "NO_RESULT"

            confidence = 0.0

            evidence = None

        else:

            predicted_grounded = (
                claim_result["grounded"]
            )

            nli_label = (
                claim_result["nli_label"]
            )

            confidence = (
                claim_result["nli_confidence"]
            )

            evidence = (
                claim_result["evidence"]
            )


        expected_grounded = (
            case["expected_grounded"]
        )


        # --------------------------------------------
        # Classification
        # --------------------------------------------

        if (
            predicted_grounded
            and expected_grounded
        ):

            true_positive += 1

            status = "CORRECT"


        elif (
            not predicted_grounded
            and not expected_grounded
        ):

            true_negative += 1

            status = "CORRECT"


        elif (
            predicted_grounded
            and not expected_grounded
        ):

            false_positive += 1

            status = "WRONG"


        else:

            false_negative += 1

            status = "WRONG"


        # --------------------------------------------
        # Print result
        # --------------------------------------------

        print(
            f"\nResult: "
            f"{status}"
        )


        print(
            f"Expected : "
            f"{'GROUNDED' if expected_grounded else 'UNGROUNDED'}"
        )

        print(
            f"Predicted: "
            f"{'GROUNDED' if predicted_grounded else 'UNGROUNDED'}"
        )

        print(
            f"NLI Label: "
            f"{nli_label}"
        )

        print(
            f"Confidence: "
            f"{confidence:.4f}"
        )


        # --------------------------------------------
        # Evidence provenance
        # --------------------------------------------

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
    # METRICS
    # ==================================================

    total = len(
        TEST_CASES
    )


    precision = (

        true_positive
        /
        (
            true_positive
            + false_positive
        )

        if (
            true_positive
            + false_positive
        ) > 0

        else 0.0
    )


    recall = (

        true_positive
        /
        (
            true_positive
            + false_negative
        )

        if (
            true_positive
            + false_negative
        ) > 0

        else 0.0
    )


    f1 = (

        2
        * precision
        * recall
        /
        (
            precision
            + recall
        )

        if (
            precision
            + recall
        ) > 0

        else 0.0
    )


    accuracy = (

        true_positive
        + true_negative
    ) / total


    # ==================================================
    # SUMMARY
    # ==================================================

    print("\n")

    print("=" * 70)

    print(
        "ADVERSARIAL GROUNDEDNESS SUMMARY"
    )

    print("=" * 70)


    print(
        f"\nTotal Cases        : "
        f"{total}"
    )

    print(
        f"True Positives     : "
        f"{true_positive}"
    )

    print(
        f"True Negatives     : "
        f"{true_negative}"
    )

    print(
        f"False Positives    : "
        f"{false_positive}"
    )

    print(
        f"False Negatives    : "
        f"{false_negative}"
    )


    print(
        f"\nPrecision          : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall             : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score           : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"Accuracy           : "
        f"{accuracy * 100:.2f}%"
    )


    print("\n")

    print("=" * 70)

    print(
        "ADVERSARIAL GROUNDEDNESS EVALUATION COMPLETE"
    )

    print("=" * 70)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    main()