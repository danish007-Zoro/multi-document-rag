import os
import sys
import numpy as np


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

from embeddings import load_embedding_model

from groundedness_test_data import (
    GROUNDEDNESS_TEST_CASES
)


# ==================================================
# CONFIGURATION
# ==================================================

THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]


# ==================================================
# SIMILARITY
# ==================================================

def calculate_similarity(
    model,
    answer,
    context
):
    """
    Calculate cosine similarity between the answer
    and the provided context.

    Embeddings are normalized so the dot product
    represents cosine similarity.
    """

    answer_embedding = model.encode(
        [answer],
        normalize_embeddings=True
    )

    context_embedding = model.encode(
        [context],
        normalize_embeddings=True
    )

    similarity = np.dot(
        answer_embedding[0],
        context_embedding[0]
    )

    return float(
        similarity
    )


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(
    cases,
    similarities,
    threshold
):

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0


    for case, similarity in zip(
        cases,
        similarities
    ):

        predicted = (
            similarity >= threshold
        )

        actual = case["grounded"]


        if predicted and actual:

            true_positive += 1

        elif not predicted and not actual:

            true_negative += 1

        elif predicted and not actual:

            false_positive += 1

        elif not predicted and actual:

            false_negative += 1


    # ------------------------------------------------
    # Precision
    # ------------------------------------------------

    if (
        true_positive
        + false_positive
        > 0
    ):

        precision = (
            true_positive
            /
            (
                true_positive
                + false_positive
            )
        )

    else:

        precision = 0.0


    # ------------------------------------------------
    # Recall
    # ------------------------------------------------

    if (
        true_positive
        + false_negative
        > 0
    ):

        recall = (
            true_positive
            /
            (
                true_positive
                + false_negative
            )
        )

    else:

        recall = 0.0


    # ------------------------------------------------
    # F1
    # ------------------------------------------------

    if (
        precision
        + recall
        > 0
    ):

        f1 = (
            2
            * precision
            * recall
            /
            (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0


    # ------------------------------------------------
    # Accuracy
    # ------------------------------------------------

    total = len(cases)

    accuracy = (
        (
            true_positive
            + true_negative
        )
        / total
        if total > 0
        else 0.0
    )


    return {
        "true_positive":
            true_positive,

        "true_negative":
            true_negative,

        "false_positive":
            false_positive,

        "false_negative":
            false_negative,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "accuracy":
            accuracy
    }


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)
    print(
        "GROUNDEDNESS THRESHOLD EVALUATION"
    )
    print("=" * 70)


    # ------------------------------------------------
    # Load model
    # ------------------------------------------------

    print("\nLoading embedding model...")

    model = load_embedding_model()


    # ------------------------------------------------
    # Calculate similarities
    # ------------------------------------------------

    print("\nCalculating similarities...")

    similarities = []


    for number, case in enumerate(
        GROUNDEDNESS_TEST_CASES,
        start=1
    ):

        similarity = calculate_similarity(
            model,
            case["answer"],
            case["context"]
        )

        similarities.append(
            similarity
        )

        label = (
            "GROUNDED"
            if case["grounded"]
            else "UNGROUNDED"
        )

        print(
            f"[{number:02d}] "
            f"{label:<12} "
            f"Similarity: "
            f"{similarity:.4f}"
        )


    # ------------------------------------------------
    # Threshold evaluation
    # ------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "THRESHOLD RESULTS"
    )
    print("=" * 70)

    print(
        "\nThreshold   Precision   Recall   "
        "F1       Accuracy   FP   FN"
    )

    print(
        "-" * 70
    )


    threshold_results = []


    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            GROUNDEDNESS_TEST_CASES,
            similarities,
            threshold
        )

        threshold_results.append(
            (
                threshold,
                metrics
            )
        )

        print(
            f"{threshold:<11.2f}"
            f"{metrics['precision'] * 100:>8.2f}%"
            f"{metrics['recall'] * 100:>10.2f}%"
            f"{metrics['f1'] * 100:>9.2f}%"
            f"{metrics['accuracy'] * 100:>11.2f}%"
            f"{metrics['false_positive']:>5}"
            f"{metrics['false_negative']:>5}"
        )


    # ------------------------------------------------
    # Best threshold
    # ------------------------------------------------

    best_threshold, best_metrics = max(
        threshold_results,
        key=lambda item: (
            item[1]["f1"],
            item[1]["accuracy"]
        )
    )


    print("\n")
    print("=" * 70)
    print(
        "BEST THRESHOLD BY F1"
    )
    print("=" * 70)

    print(
        f"\nThreshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Precision : "
        f"{best_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{best_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score  : "
        f"{best_metrics['f1'] * 100:.2f}%"
    )

    print(
        f"Accuracy  : "
        f"{best_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"FP        : "
        f"{best_metrics['false_positive']}"
    )

    print(
        f"FN        : "
        f"{best_metrics['false_negative']}"
    )


    # ------------------------------------------------
    # Detailed examples
    # ------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "SIMILARITY DISTRIBUTION"
    )
    print("=" * 70)


    grounded_scores = [
        similarity
        for case, similarity
        in zip(
            GROUNDEDNESS_TEST_CASES,
            similarities
        )
        if case["grounded"]
    ]

    ungrounded_scores = [
        similarity
        for case, similarity
        in zip(
            GROUNDEDNESS_TEST_CASES,
            similarities
        )
        if not case["grounded"]
    ]


    print(
        f"\nGrounded minimum : "
        f"{min(grounded_scores):.4f}"
    )

    print(
        f"Grounded maximum : "
        f"{max(grounded_scores):.4f}"
    )

    print(
        f"Ungrounded minimum : "
        f"{min(ungrounded_scores):.4f}"
    )

    print(
        f"Ungrounded maximum : "
        f"{max(ungrounded_scores):.4f}"
    )


    print("\n")
    print("=" * 70)
    print(
        "GROUNDEDNESS THRESHOLD "
        "EVALUATION COMPLETE"
    )
    print("=" * 70)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()