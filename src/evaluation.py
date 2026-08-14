# ==================================================
# EVALUATION DATASET
# ==================================================

TEST_QUERIES = [

    # ------------------------------------------------
    # RELEVANT — Direct / General
    # ------------------------------------------------

    {
        "query": "What is Artificial Intelligence?",
        "relevant": True,
        "expected_sources": [
            {
                "document": "Sample1.pdf",
                "chunk_id": 1
            }
        ]
    },

    {
        "query": "What is AI?",
        "relevant": True,
        "expected_sources": [
            {
                "document": "Sample1.pdf",
                "chunk_id": 1
            }
        ]
    },

    {
        "query": "How does Artificial Intelligence help machines solve problems?",
        "relevant": True,
        "expected_sources": [
            {
                "document": "Sample1.pdf",
                "chunk_id": 1
            }
        ]
    },

    # ------------------------------------------------
    # HUMAN-LIKE BEHAVIOUR / LEARNING
    # ------------------------------------------------

    {
        "query": "How can machines behave intelligently like humans?",
        "relevant": True,
        "expected_sources": [
            {
                "document": "Sample1.pdf",
                "chunk_id": 1
            },
            {
                "document": "Sample1.pdf",
                "chunk_id": 11
            }
        ]
    },

    {
        "query": "How does AI improve by learning from mistakes?",
        "relevant": True,
        "expected_sources": [
            {
                "document": "Sample1.pdf",
                "chunk_id": 2
            },
            {
                "document": "Sample1.pdf",
                "chunk_id": 3
            }
        ]
    },

    {
        "query": "How is learning from mistakes compared to playing chess?",
        "relevant": True,
        "expected_sources": [
            {
                "document": "Sample1.pdf",
                "chunk_id": 2
            },
            {
                "document": "Sample1.pdf",
                "chunk_id": 3
            }
        ]
    },

    # ------------------------------------------------
    # AI / MACHINE LEARNING
    # ------------------------------------------------

    {
        "query": "How are Artificial Intelligence and Machine Learning related?",
        "relevant": True,
        "expected_sources": [
            {
                "document": "Sample1.pdf",
                "chunk_id": 11
            }
        ]
    },

    # ------------------------------------------------
    # AI HISTORY
    # ------------------------------------------------

    {
        "query": "When was the Turing test invented?",
        "relevant": True,
        "expected_sources": [
            {
                "document": "Sample1.pdf",
                "chunk_id": 9
            }
        ]
    },

    {
        "query": "When was ELIZA created?",
        "relevant": True,
        "expected_sources": [
            {
                "document": "Sample1.pdf",
                "chunk_id": 9
            }
        ]
    },

    {
        "query": "When was Siri announced?",
        "relevant": True,
        "expected_sources": [
            {
                "document": "Sample1.pdf",
                "chunk_id": 10
            }
        ]
    },

    {
        "query": "When was OpenAI founded?",
        "relevant": True,
        "expected_sources": [
            {
                "document": "Sample1.pdf",
                "chunk_id": 10
            }
        ]
    },

    {
        "query": "What computer defeated a world chess champion?",
        "relevant": True,
        "expected_sources": [
            {
                "document": "Sample1.pdf",
                "chunk_id": 9
            },
            {
                "document": "Sample1.pdf",
                "chunk_id": 10
            }
        ]
    },

    # ------------------------------------------------
    # IRRELEVANT
    # ------------------------------------------------

    {
        "query": "What is the capital of France?",
        "relevant": False,
        "expected_sources": []
    },

    {
        "query": "Who is the Prime Minister of India?",
        "relevant": False,
        "expected_sources": []
    },

    {
        "query": "How do you bake a chocolate cake?",
        "relevant": False,
        "expected_sources": []
    },

    {
        "query": "What is the boiling point of water?",
        "relevant": False,
        "expected_sources": []
    },

    {
        "query": "How does a car engine work?",
        "relevant": False,
        "expected_sources": []
    },

    {
        "query": "What is the largest planet in the solar system?",
        "relevant": False,
        "expected_sources": []
    },

    {
        "query": "Who wrote Romeo and Juliet?",
        "relevant": False,
        "expected_sources": []
    },

    {
        "query": "What is the currency of Japan?",
        "relevant": False,
        "expected_sources": []
    },

    {
        "query": "How many continents are there?",
        "relevant": False,
        "expected_sources": []
    },

    {
        "query": "What is the tallest mountain in the world?",
        "relevant": False,
        "expected_sources": []
    }
]

# ==================================================
# RETRIEVAL EVALUATION
# ==================================================

def evaluate_retrieval(
    test_queries,
    model,
    index,
    chunks,
    top_k=3,
    distance_threshold=1.4
):
    """
    Run retrieval evaluation across a set of
    test queries.

    Returns:
        list:
            Individual evaluation results.
    """

    from retrieval import search

    evaluation_results = []

    for item in test_queries:

        query = item["query"]

        expected_relevant = item["relevant"]

        expected_sources = item[
            "expected_sources"
        ]

        retrieval = search(
            query,
            model,
            index,
            chunks,
            top_k=top_k,
            distance_threshold=distance_threshold
        )

        result = {
            "query": query,

            "expected_relevant":
                expected_relevant,

            "expected_sources":
                expected_sources,

            "accepted":
                retrieval["accepted"],

            "best_distance":
                retrieval["best_distance"],

            "results":
                retrieval["results"]
        }

        evaluation_results.append(
            result
        )

    return evaluation_results


# ==================================================
# THRESHOLD EVALUATION
# ==================================================

def evaluate_thresholds(
    test_queries,
    model,
    index,
    chunks,
    thresholds,
    top_k=3
):
    """
    Evaluate retrieval classification across
    multiple distance thresholds.

    Returns:
        list:
            Metrics for each threshold.
    """

    results = []

    for threshold in thresholds:

        evaluation_results = evaluate_retrieval(
            test_queries,
            model,
            index,
            chunks,
            top_k=top_k,
            distance_threshold=threshold
        )

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for result in evaluation_results:

            expected = (
                result["expected_relevant"]
            )

            accepted = (
                result["accepted"]
            )

            if expected and accepted:

                true_positive += 1

            elif not expected and not accepted:

                true_negative += 1

            elif not expected and accepted:

                false_positive += 1

            elif expected and not accepted:

                false_negative += 1

        precision = (
            true_positive /
            (
                true_positive +
                false_positive
            )
            if (
                true_positive +
                false_positive
            ) > 0
            else 0.0
        )

        recall = (
            true_positive /
            (
                true_positive +
                false_negative
            )
            if (
                true_positive +
                false_negative
            ) > 0
            else 0.0
        )

        f1_score = (
            2 * precision * recall /
            (
                precision +
                recall
            )
            if (
                precision +
                recall
            ) > 0
            else 0.0
        )

        accuracy = (
            (
                true_positive +
                true_negative
            )
            /
            len(evaluation_results)
            if evaluation_results
            else 0.0
        )

        results.append(
            {
                "threshold":
                    threshold,

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

                "f1_score":
                    f1_score,

                "accuracy":
                    accuracy
            }
        )

    return results


# ==================================================
# HIT@K
# ==================================================

def calculate_hit_rate(
    evaluation_results,
    k=3
):
    """
    Calculate Hit@K for relevant queries.

    A query is considered a hit when at least one
    expected document/chunk pair appears within
    the top-k retrieved results.
    """

    relevant_queries = 0

    hits = 0

    for result in evaluation_results:

        if not result[
            "expected_relevant"
        ]:
            continue

        relevant_queries += 1

        expected_sources = {
            (
                source["document"],
                source["chunk_id"]
            )
            for source in
            result["expected_sources"]
        }

        retrieved_sources = {
            (
                chunk.get("document"),
                chunk.get("chunk_id")
            )
            for chunk in
            result["results"][:k]
        }

        if expected_sources.intersection(
            retrieved_sources
        ):

            hits += 1

    if relevant_queries == 0:

        return 0.0

    return (
        hits /
        relevant_queries
    )


# ==================================================
# MRR
# ==================================================

def calculate_mrr(
    evaluation_results
):
    """
    Calculate Mean Reciprocal Rank.

    The first relevant document/chunk pair
    determines the reciprocal rank.
    """

    reciprocal_ranks = []

    for result in evaluation_results:

        if not result[
            "expected_relevant"
        ]:
            continue

        expected_sources = {
            (
                source["document"],
                source["chunk_id"]
            )
            for source in
            result["expected_sources"]
        }

        reciprocal_rank = 0.0

        for rank, chunk in enumerate(
            result["results"],
            start=1
        ):

            retrieved_source = (
                chunk.get("document"),
                chunk.get("chunk_id")
            )

            if (
                retrieved_source
                in expected_sources
            ):

                reciprocal_rank = (
                    1.0 / rank
                )

                break

        reciprocal_ranks.append(
            reciprocal_rank
        )

    if not reciprocal_ranks:

        return 0.0

    return (
        sum(reciprocal_ranks)
        /
        len(reciprocal_ranks)
    )


# ==================================================
# EVALUATION SUMMARY
# ==================================================

def calculate_metrics(
    evaluation_results,
    k=3
):
    """
    Calculate the main retrieval metrics.

    Returns:
        dict:
            Classification, Hit@K and MRR metrics.
    """

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for result in evaluation_results:

        expected = (
            result["expected_relevant"]
        )

        accepted = (
            result["accepted"]
        )

        if expected and accepted:

            true_positive += 1

        elif not expected and not accepted:

            true_negative += 1

        elif not expected and accepted:

            false_positive += 1

        elif expected and not accepted:

            false_negative += 1

    precision = (
        true_positive /
        (
            true_positive +
            false_positive
        )
        if (
            true_positive +
            false_positive
        ) > 0
        else 0.0
    )

    recall = (
        true_positive /
        (
            true_positive +
            false_negative
        )
        if (
            true_positive +
            false_negative
        ) > 0
        else 0.0
    )

    f1_score = (
        2 * precision * recall /
        (
            precision +
            recall
        )
        if (
            precision +
            recall
        ) > 0
        else 0.0
    )

    accuracy = (
        (
            true_positive +
            true_negative
        )
        /
        len(evaluation_results)
        if evaluation_results
        else 0.0
    )

    hit_at_k = calculate_hit_rate(
        evaluation_results,
        k=k
    )

    mrr = calculate_mrr(
        evaluation_results
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

        "f1_score":
            f1_score,

        "accuracy":
            accuracy,

        "hit_at_k":
            hit_at_k,

        "mrr":
            mrr
    }