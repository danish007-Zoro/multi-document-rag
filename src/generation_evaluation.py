# ==================================================
# GENERATION EVALUATION
# ==================================================

TEST_QUERIES = [

    # ------------------------------------------------
    # DIRECT QUESTIONS
    # ------------------------------------------------

    {
        "query": "What is Artificial Intelligence?",
        "expected_answer_type": "answer"
    },

    {
        "query": "What is Machine Learning?",
        "expected_answer_type": "answer"
    },

    {
        "query": "When was the Turing test invented?",
        "expected_answer_type": "answer"
    },

    {
        "query": "When was ELIZA created?",
        "expected_answer_type": "answer"
    },

    {
        "query": "When was Siri announced?",
        "expected_answer_type": "answer"
    },

    {
        "query": "When was OpenAI founded?",
        "expected_answer_type": "answer"
    },

    {
        "query": "What are the applications of machine learning?",
        "expected_answer_type": "answer"
    },

    {
        "query": "How is machine learning used in gaming?",
        "expected_answer_type": "answer"
    },


    # ------------------------------------------------
    # IRRELEVANT QUESTIONS
    # ------------------------------------------------

    {
        "query": "What is the capital of France?",
        "expected_answer_type": "rejection"
    },

    {
        "query": "What is the boiling point of water?",
        "expected_answer_type": "rejection"
    },

    {
        "query": "Who wrote Romeo and Juliet?",
        "expected_answer_type": "rejection"
    },

    {
        "query": "How does a car engine work?",
        "expected_answer_type": "rejection"
    }
]


# ==================================================
# EXPECTED REJECTION
# ==================================================

EXPECTED_REJECTION = (
    "I could not find the answer "
    "in the provided documents."
)


# ==================================================
# EVALUATE GENERATED ANSWER
# ==================================================

def evaluate_answer(
    result,
    expected_answer_type
):
    """
    Determine whether the generated answer follows
    the expected behavior.

    Relevant queries:
        Must be accepted AND must not return the
        standard rejection response.

    Irrelevant queries:
        Must be rejected AND must return the
        standard rejection response.
    """

    answer = (
        result["answer"]
        .strip()
    )

    if expected_answer_type == "answer":

        return (
            result["accepted"]
            and
            answer != EXPECTED_REJECTION
            and
            len(answer) > 0
        )

    if expected_answer_type == "rejection":

        return (
            not result["accepted"]
            and
            answer == EXPECTED_REJECTION
        )

    return False