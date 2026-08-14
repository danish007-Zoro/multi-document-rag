import os
import sys
import numpy as np


# ==================================================
# PATH SETUP
# ==================================================

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
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

from claim_groundedness_test_data import (
    CLAIM_GROUNDEDNESS_TEST_CASES
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
    claim,
    context
):

    claim_embedding = model.encode(
        [claim],
        normalize_embeddings=True
    )

    context_embedding = model.encode(
        [context],
        normalize_embeddings=True
    )

    similarity = np.dot(
        claim_embedding[0],
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

    tp = 0
    tn = 0
    fp = 0
    fn = 0


    for case, similarity in zip(
        cases,
        similarities
    ):

        predicted = (
            similarity >= threshold
        )

        actual = case["grounded"]


        if predicted and actual:

            tp += 1

        elif not predicted and not actual:

            tn += 1

        elif predicted and not actual:

            fp += 1

        else:

            fn += 1


    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / len(cases)
    )


    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)
    print(
        "CLAIM-LEVEL GROUNDEDNESS EVALUATION"
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

    print("\nCalculating claim similarities...")

    similarities = []


    for number, case in enumerate(
        CLAIM_GROUNDEDNESS_TEST_CASES,
        start=1
    ):

        similarity = calculate_similarity(
            model,
            case["claim"],
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
    # Threshold results
    # ------------------------------------------------

    print("\n")
    print("=" * 70)
    print("THRESHOLD RESULTS")
    print("=" * 70)

    print(
        "\nThreshold   Precision   Recall   "
        "F1       Accuracy   FP   FN"
    )

    print(
        "-" * 70
    )


    results = []


    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            CLAIM_GROUNDEDNESS_TEST_CASES,
            similarities,
            threshold
        )

        results.append(
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
            f"{metrics['fp']:>5}"
            f"{metrics['fn']:>5}"
        )


    # ------------------------------------------------
    # Best threshold
    # ------------------------------------------------

    best_threshold, best_metrics = max(
        results,
        key=lambda item: (
            item[1]["f1"],
            item[1]["accuracy"]
        )
    )


    print("\n")
    print("=" * 70)
    print("BEST THRESHOLD BY F1")
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
        f"{best_metrics['fp']}"
    )

    print(
        f"FN        : "
        f"{best_metrics['fn']}"
    )


    # ------------------------------------------------
    # Distribution
    # ------------------------------------------------

    grounded_scores = [
        score
        for case, score
        in zip(
            CLAIM_GROUNDEDNESS_TEST_CASES,
            similarities
        )
        if case["grounded"]
    ]

    ungrounded_scores = [
        score
        for case, score
        in zip(
            CLAIM_GROUNDEDNESS_TEST_CASES,
            similarities
        )
        if not case["grounded"]
    ]


    print("\n")
    print("=" * 70)
    print("SIMILARITY DISTRIBUTION")
    print("=" * 70)

    print(
        f"\nGrounded minimum   : "
        f"{min(grounded_scores):.4f}"
    )

    print(
        f"Grounded maximum   : "
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


    # ------------------------------------------------
    # Detailed classification at best threshold
    # ------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        f"DETAILED RESULTS "
        f"AT THRESHOLD {best_threshold:.2f}"
    )
    print("=" * 70)


    for number, (
        case,
        similarity
    ) in enumerate(
        zip(
            CLAIM_GROUNDEDNESS_TEST_CASES,
            similarities
        ),
        start=1
    ):

        predicted = (
            similarity >= best_threshold
        )

        actual = case["grounded"]

        if predicted == actual:

            status = "CORRECT"

        else:

            status = "WRONG"


        actual_label = (
            "GROUNDED"
            if actual
            else "UNGROUNDED"
        )

        predicted_label = (
            "GROUNDED"
            if predicted
            else "UNGROUNDED"
        )


        print(
            f"\n[{number:02d}] {status}"
        )

        print(
            f"Actual    : "
            f"{actual_label}"
        )

        print(
            f"Predicted : "
            f"{predicted_label}"
        )

        print(
            f"Similarity: "
            f"{similarity:.4f}"
        )

        print(
            f"Claim     : "
            f"{case['claim']}"
        )


    print("\n")
    print("=" * 70)
    print(
        "CLAIM-LEVEL GROUNDEDNESS "
        "EVALUATION COMPLETE"
    )
    print("=" * 70)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()