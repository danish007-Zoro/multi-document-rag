from nli_groundedness import (
    NLIGroundednessEvaluator
)


# ==================================================
# TEST CASES
# ==================================================

TEST_CASES = [

    {
        "name":
            "Direct entailment",

        "premise":
            "Siri was announced as a digital "
            "assistant by Apple in 2011.",

        "hypothesis":
            "Siri was announced by Apple in 2011."
    },


    {
        "name":
            "Contradiction",

        "premise":
            "Siri was announced as a digital "
            "assistant by Apple in 2011.",

        "hypothesis":
            "Siri was announced by Apple in 2008."
    },


    {
        "name":
            "Wrong person",

        "premise":
            "Elon Musk and some others founded "
            "OpenAI in 2015.",

        "hypothesis":
            "OpenAI was founded by Bill Gates in 2015."
    },


    {
        "name":
            "Neutral",

        "premise":
            "Siri was announced as a digital "
            "assistant by Apple in 2011.",

        "hypothesis":
            "Siri was announced at Apple's headquarters."
    }
]


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)
    print(
        "NLI MODEL SANITY TEST"
    )
    print("=" * 70)


    evaluator = (
        NLIGroundednessEvaluator()
    )


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
            f"{case['name']}"
        )

        print(
            f"\nPremise:\n"
            f"{case['premise']}"
        )

        print(
            f"\nHypothesis:\n"
            f"{case['hypothesis']}"
        )


        result = evaluator.predict(
            case["premise"],
            case["hypothesis"]
        )


        print(
            f"\nPredicted Label : "
            f"{result['label']}"
        )


        print(
            "\nScores:"
        )

        for label, score in (
            result["scores"].items()
        ):

            print(
                f"  {label:<15}: "
                f"{score:.4f}"
            )


    print("\n")
    print("=" * 70)
    print(
        "NLI MODEL SANITY TEST COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()