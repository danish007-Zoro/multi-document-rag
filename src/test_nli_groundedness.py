import os
import sys


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

from nli_groundedness import (
    NLIGroundednessEvaluator
)

from claim_groundedness_test_data import (
    CLAIM_GROUNDEDNESS_TEST_CASES
)


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)
    print(
        "NLI GROUNDEDNESS EVALUATION"
    )
    print("=" * 70)


    # ------------------------------------------------
    # Load evaluator
    # ------------------------------------------------

    print("\nLoading NLI model...")

    evaluator = (
        NLIGroundednessEvaluator()
    )


    # ------------------------------------------------
    # Counters
    # ------------------------------------------------

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    detailed_results = []


    # ------------------------------------------------
    # Evaluate claims
    # ------------------------------------------------

    print("\nEvaluating claims...")


    for number, case in enumerate(
        CLAIM_GROUNDEDNESS_TEST_CASES,
        start=1
    ):

        result = evaluator.evaluate_claim(
            case["claim"],
            [case["context"]]
        )


        predicted_grounded = (
            result["grounded"]
        )

        actual_grounded = (
            case["grounded"]
        )


        # --------------------------------------------
        # Classification
        # --------------------------------------------

        if (
            predicted_grounded
            and actual_grounded
        ):

            true_positive += 1

            status = "CORRECT"


        elif (
            not predicted_grounded
            and not actual_grounded
        ):

            true_negative += 1

            status = "CORRECT"


        elif (
            predicted_grounded
            and not actual_grounded
        ):

            false_positive += 1

            status = "WRONG"


        else:

            false_negative += 1

            status = "WRONG"


        # --------------------------------------------
        # Store result
        # --------------------------------------------

        detailed_results.append(
            {
                "number": number,
                "status": status,
                "actual": actual_grounded,
                "predicted": predicted_grounded,
                "label": result["label"],
                "confidence": result["confidence"],
                "claim": case["claim"],
                "context": case["context"]
            }
        )


    # ==================================================
    # METRICS
    # ==================================================

    total = len(
        CLAIM_GROUNDEDNESS_TEST_CASES
    )


    grounded_count = sum(
        1
        for case in CLAIM_GROUNDEDNESS_TEST_CASES
        if case["grounded"]
    )


    ungrounded_count = sum(
        1
        for case in CLAIM_GROUNDEDNESS_TEST_CASES
        if not case["grounded"]
    )


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
    # F1 Score
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

    if total > 0:

        accuracy = (
            true_positive
            + true_negative
        ) / total

    else:

        accuracy = 0.0


    # ==================================================
    # SUMMARY
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "NLI GROUNDEDNESS SUMMARY"
    )
    print("=" * 70)


    print(
        f"\nTotal Claims        : "
        f"{total}"
    )

    print(
        f"Grounded Claims     : "
        f"{grounded_count}"
    )

    print(
        f"Ungrounded Claims   : "
        f"{ungrounded_count}"
    )


    # ==================================================
    # CLASSIFICATION METRICS
    # ==================================================

    print("\n")
    print(
        "CLASSIFICATION METRICS"
    )

    print(
        "-" * 70
    )


    print(
        f"True Positives      : "
        f"{true_positive}"
    )

    print(
        f"True Negatives      : "
        f"{true_negative}"
    )

    print(
        f"False Positives     : "
        f"{false_positive}"
    )

    print(
        f"False Negatives     : "
        f"{false_negative}"
    )


    print(
        f"Precision           : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall              : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score            : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"Accuracy            : "
        f"{accuracy * 100:.2f}%"
    )


    # ==================================================
    # DETAILED RESULTS
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "DETAILED RESULTS"
    )
    print("=" * 70)


    for item in detailed_results:

        print("\n")

        print(
            f"[{item['number']:02d}] "
            f"{item['status']}"
        )


        # --------------------------------------------
        # Actual label
        # --------------------------------------------

        if item["actual"]:

            actual_label = "GROUNDED"

        else:

            actual_label = "UNGROUNDED"


        # --------------------------------------------
        # Predicted label
        # --------------------------------------------

        if item["predicted"]:

            predicted_label = "GROUNDED"

        else:

            predicted_label = "UNGROUNDED"


        print(
            f"Actual    : "
            f"{actual_label}"
        )

        print(
            f"Predicted : "
            f"{predicted_label}"
        )

        print(
            f"NLI Label : "
            f"{item['label']}"
        )

        print(
            f"Confidence: "
            f"{item['confidence']:.4f}"
        )

        print(
            f"Claim     : "
            f"{item['claim']}"
        )


    # ==================================================
    # COMPLETE
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "NLI GROUNDEDNESS EVALUATION COMPLETE"
    )
    print("=" * 70)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()