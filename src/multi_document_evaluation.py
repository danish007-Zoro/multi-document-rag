# ==================================================
# MULTI-DOCUMENT RAG EVALUATION
# ==================================================


# ==================================================
# TEST QUERIES
# ==================================================

TEST_QUERIES = [

    # ==================================================
    # AI — SAMPLE 1 / SAMPLE 2
    #
    # Sample1 and Sample2 contain duplicate content.
    # Either document is therefore a valid source.
    # ==================================================

    {
        "query": "What is Artificial Intelligence?",
        "relevant": True,
        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf"
        ],
        "expected_sources": [
            ("Sample1.pdf", 1),
            ("Sample2.pdf", 1)
        ]
    },

    {
        "query": "How does AI learn from mistakes?",
        "relevant": True,
        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf"
        ],
        "expected_sources": [
            ("Sample1.pdf", 2),
            ("Sample2.pdf", 2)
        ]
    },

    {
        "query": "When was the Turing test invented?",
        "relevant": True,
        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf"
        ],
        "expected_sources": [
            ("Sample1.pdf", 9),
            ("Sample2.pdf", 9)
        ]
    },

    {
        "query": "When was ELIZA created?",
        "relevant": True,
        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf"
        ],
        "expected_sources": [
            ("Sample1.pdf", 10),
            ("Sample2.pdf", 10)
        ]
    },

    {
        "query": "When was Siri announced?",
        "relevant": True,
        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf"
        ],
        "expected_sources": [
            ("Sample1.pdf", 11),
            ("Sample2.pdf", 11)
        ]
    },

    {
        "query": "When was OpenAI founded?",
        "relevant": True,
        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf"
        ],
        "expected_sources": [
            ("Sample1.pdf", 11),
            ("Sample2.pdf", 11)
        ]
    },

    {
        "query": "What are the concerns about Artificial Intelligence?",
        "relevant": True,
        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf"
        ],
        "expected_sources": [
            ("Sample1.pdf", 24),
            ("Sample2.pdf", 24)
        ]
    },


    # ==================================================
    # MACHINE LEARNING — SAMPLE 3
    # ==================================================

    {
        "query": "What is Machine Learning?",
        "relevant": True,
        "expected_documents": [
            "Sample3.pdf"
        ],
        "expected_sources": [
            ("Sample3.pdf", 11)
        ]
    },

    {
        "query": "How does machine learning work?",
        "relevant": True,
        "expected_documents": [
            "Sample3.pdf"
        ],
        "expected_sources": [
            ("Sample3.pdf", 22)
        ]
    },

    {
        "query": "What is supervised learning?",
        "relevant": True,
        "expected_documents": [
            "Sample3.pdf"
        ],
        "expected_sources": [
            ("Sample3.pdf", 27)
        ]
    },

    {
        "query": "What is unsupervised learning?",
        "relevant": True,
        "expected_documents": [
            "Sample3.pdf"
        ],
        "expected_sources": [
            ("Sample1.pdf", 16),
            ("Sample2.pdf", 16),
            ("Sample3.pdf", 30)
        ]
    },

    {
        "query": "What are applications of machine learning?",
        "relevant": True,
        "expected_documents": [
            "Sample3.pdf"
        ],
        "expected_sources": [
            ("Sample3.pdf", 8),
            ("Sample3.pdf", 15),
            ("Sample3.pdf", 58)
        ]
    },

    {
        "query": "How is machine learning used for fraud detection?",
        "relevant": True,
        "expected_documents": [
            "Sample3.pdf"
        ],
        "expected_sources": [
            ("Sample3.pdf", 42)
        ]
    },

    {
        "query": "How is machine learning used in gaming?",
        "relevant": True,
        "expected_documents": [
            "Sample3.pdf"
        ],
        "expected_sources": [
            ("Sample3.pdf", 54)
        ]
    },

    {
        "query": "What are the challenges of machine learning?",
        "relevant": True,
        "expected_documents": [
            "Sample3.pdf"
        ],
        "expected_sources": [
            ("Sample3.pdf", 60)
        ]
    },


    # ==================================================
    # CROSS-DOCUMENT
    # ==================================================

    {
        "query": "How are Artificial Intelligence and Machine Learning related?",
        "relevant": True,
        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf",
            "Sample3.pdf"
        ],
        "expected_sources": [
            ("Sample1.pdf", 12),
            ("Sample2.pdf", 12),
            ("Sample3.pdf", 10)
        ]
    },


    # ==================================================
    # IRRELEVANT
    # ==================================================

    {
        "query": "What is the capital of France?",
        "relevant": False,
        "expected_documents": [],
        "expected_sources": []
    },

    {
        "query": "Who is the Prime Minister of India?",
        "relevant": False,
        "expected_documents": [],
        "expected_sources": []
    },

    {
        "query": "How do you bake a chocolate cake?",
        "relevant": False,
        "expected_documents": [],
        "expected_sources": []
    },

    {
        "query": "What is the boiling point of water?",
        "relevant": False,
        "expected_documents": [],
        "expected_sources": []
    },

    {
        "query": "How does a car engine work?",
        "relevant": False,
        "expected_documents": [],
        "expected_sources": []
    },

    {
        "query": "What is the largest planet in the solar system?",
        "relevant": False,
        "expected_documents": [],
        "expected_sources": []
    },

    {
        "query": "Who wrote Romeo and Juliet?",
        "relevant": False,
        "expected_documents": [],
        "expected_sources": []
    },

    {
        "query": "What is the currency of Japan?",
        "relevant": False,
        "expected_documents": [],
        "expected_sources": []
    }
]


# ==================================================
# DOCUMENT-LEVEL RETRIEVAL EVALUATION
# ==================================================

def evaluate_document_attribution(
    test_queries,
    model,
    index,
    chunks,
    top_k=3,
    distance_threshold=1.4
):
    """
    Evaluate document-level retrieval.

    For relevant queries:
        A source hit occurs when at least one
        expected document appears in the top-k results.

    For irrelevant queries:
        The query passes when retrieval is rejected.
    """

    from retrieval import search

    evaluation_results = []

    for item in test_queries:

        query = item["query"]

        expected_relevant = (
            item["relevant"]
        )

        expected_documents = set(
            item["expected_documents"]
        )

        expected_sources = set(
            tuple(source)
            for source in item.get(
                "expected_sources",
                []
            )
        )

        retrieval = search(
            query,
            model,
            index,
            chunks,
            top_k=top_k,
            distance_threshold=distance_threshold
        )

        retrieved_documents = []

        for chunk in retrieval["results"]:

            document = chunk.get(
                "document",
                "Unknown"
            )

            if document not in retrieved_documents:

                retrieved_documents.append(
                    document
                )

        # ------------------------------------------
        # Source hit
        # ------------------------------------------

        if expected_relevant:

            retrieved_set = set(
                retrieved_documents
            )

            source_hit = bool(
                expected_documents.intersection(
                    retrieved_set
                )
            )

        else:

            source_hit = (
                not retrieval["accepted"]
            )

        # ------------------------------------------
        # Store result
        # ------------------------------------------

        evaluation_results.append(
            {
                "query": query,

                "expected_relevant":
                    expected_relevant,

                "expected_documents":
                    sorted(
                        expected_documents
                    ),

                "expected_sources":
                    sorted(
                        expected_sources
                    ),

                "accepted":
                    retrieval["accepted"],

                "best_distance":
                    retrieval["best_distance"],

                "retrieved_documents":
                    retrieved_documents,

                "results":
                    retrieval["results"],

                "source_hit":
                    source_hit
            }
        )

    return evaluation_results


# ==================================================
# DOCUMENT SOURCE HIT@K
# ==================================================

def calculate_document_hit_rate(
    evaluation_results,
    k=3
):
    """
    Calculate document-level Hit@K.

    For relevant queries, a hit occurs when at least
    one expected document appears in the top-k results.

    Irrelevant queries are excluded from this metric.
    """

    relevant_queries = 0
    hits = 0

    for result in evaluation_results:

        if not result[
            "expected_relevant"
        ]:
            continue

        relevant_queries += 1

        expected_documents = set(
            result["expected_documents"]
        )

        retrieved_documents = set()

        for chunk in result[
            "results"
        ][:k]:

            document = chunk.get(
                "document"
            )

            if document is not None:

                retrieved_documents.add(
                    document
                )

        if expected_documents.intersection(
            retrieved_documents
        ):

            hits += 1

    if relevant_queries == 0:

        return 0.0

    return (
        hits /
        relevant_queries
    )


# ==================================================
# DOCUMENT MRR
# ==================================================

def calculate_document_mrr(
    evaluation_results
):
    """
    Calculate document-level Mean Reciprocal Rank.

    For each relevant query, the rank of the first
    retrieved chunk belonging to an expected document
    is used.
    """

    reciprocal_ranks = []

    for result in evaluation_results:

        if not result[
            "expected_relevant"
        ]:
            continue

        expected_documents = set(
            result["expected_documents"]
        )

        reciprocal_rank = 0.0

        for rank, chunk in enumerate(
            result["results"],
            start=1
        ):

            document = chunk.get(
                "document"
            )

            if document in expected_documents:

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
# DOCUMENT HIT@1
# ==================================================

def calculate_document_hit_at_1(
    evaluation_results
):
    """
    Calculate document-level Hit@1.

    A relevant query is a hit if an expected document
    appears as the FIRST retrieved result.

    Irrelevant queries are excluded.
    """

    relevant_queries = 0
    hits = 0

    for result in evaluation_results:

        if not result["expected_relevant"]:
            continue

        relevant_queries += 1

        expected_documents = set(
            result["expected_documents"]
        )

        results = result["results"]

        if not results:
            continue

        first_document = results[0].get(
            "document"
        )

        if first_document in expected_documents:
            hits += 1

    if relevant_queries == 0:
        return 0.0

    return (
        hits /
        relevant_queries
    )


# ==================================================
# DOCUMENT RANK DISTRIBUTION
# ==================================================

def calculate_document_rank_distribution(
    evaluation_results
):
    """
    Determine the rank of the first retrieved chunk
    belonging to an expected document.

    Returns:

        Rank 1
        Rank 2
        Rank 3
        Not retrieved
    """

    distribution = {
        1: 0,
        2: 0,
        3: 0,
        "not_retrieved": 0
    }

    for result in evaluation_results:

        if not result["expected_relevant"]:
            continue

        expected_documents = set(
            result["expected_documents"]
        )

        first_relevant_rank = None

        for rank, chunk in enumerate(
            result["results"],
            start=1
        ):

            document = chunk.get(
                "document"
            )

            if document in expected_documents:

                first_relevant_rank = rank

                break

        if first_relevant_rank in distribution:

            distribution[
                first_relevant_rank
            ] += 1

        else:

            distribution[
                "not_retrieved"
            ] += 1

    return distribution


# ==================================================
# CHUNK-LEVEL SOURCE MATCHING
# ==================================================

def is_expected_source(
    chunk,
    expected_sources
):
    """
    Check whether a retrieved chunk matches one of
    the expected document + chunk combinations.

    Expected source format:

        ("Sample3.pdf", 30)
    """

    document = chunk.get(
        "document"
    )

    chunk_id = chunk.get(
        "chunk_id"
    )

    return (
        document,
        chunk_id
    ) in expected_sources


# ==================================================
# CHUNK HIT@K
# ==================================================

def calculate_chunk_hit_rate(
    evaluation_results,
    k=3
):
    """
    Calculate exact source-level Hit@K.

    A hit occurs when at least one expected
    document + chunk pair appears in the top-k.

    Only relevant queries are included.
    """

    relevant_queries = 0
    hits = 0

    for result in evaluation_results:

        if not result[
            "expected_relevant"
        ]:
            continue

        relevant_queries += 1

        expected_sources = set(
            tuple(source)
            for source in result.get(
                "expected_sources",
                []
            )
        )

        if not expected_sources:
            continue

        retrieved_results = result[
            "results"
        ][:k]

        for chunk in retrieved_results:

            if is_expected_source(
                chunk,
                expected_sources
            ):

                hits += 1
                break

    if relevant_queries == 0:

        return 0.0

    return (
        hits /
        relevant_queries
    )


# ==================================================
# CHUNK HIT@1
# ==================================================

def calculate_chunk_hit_at_1(
    evaluation_results
):
    """
    Calculate exact source-level Hit@1.

    The first retrieved chunk must match one of
    the expected document + chunk pairs.
    """

    relevant_queries = 0
    hits = 0

    for result in evaluation_results:

        if not result[
            "expected_relevant"
        ]:
            continue

        relevant_queries += 1

        expected_sources = set(
            tuple(source)
            for source in result.get(
                "expected_sources",
                []
            )
        )

        results = result[
            "results"
        ]

        if not results:
            continue

        first_chunk = results[0]

        if is_expected_source(
            first_chunk,
            expected_sources
        ):

            hits += 1

    if relevant_queries == 0:

        return 0.0

    return (
        hits /
        relevant_queries
    )


# ==================================================
# CHUNK MRR
# ==================================================

def calculate_chunk_mrr(
    evaluation_results
):
    """
    Calculate exact source-level Mean Reciprocal Rank.

    For each relevant query, the rank of the first
    expected document + chunk pair is used.
    """

    reciprocal_ranks = []

    for result in evaluation_results:

        if not result[
            "expected_relevant"
        ]:
            continue

        expected_sources = set(
            tuple(source)
            for source in result.get(
                "expected_sources",
                []
            )
        )

        reciprocal_rank = 0.0

        for rank, chunk in enumerate(
            result["results"],
            start=1
        ):

            if is_expected_source(
                chunk,
                expected_sources
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
# CHUNK RANK DISTRIBUTION
# ==================================================

def calculate_chunk_rank_distribution(
    evaluation_results
):
    """
    Determine where the first expected chunk
    appears in the retrieved results.
    """

    distribution = {
        1: 0,
        2: 0,
        3: 0,
        "not_retrieved": 0
    }

    for result in evaluation_results:

        if not result[
            "expected_relevant"
        ]:
            continue

        expected_sources = set(
            tuple(source)
            for source in result.get(
                "expected_sources",
                []
            )
        )

        first_relevant_rank = None

        for rank, chunk in enumerate(
            result["results"],
            start=1
        ):

            if is_expected_source(
                chunk,
                expected_sources
            ):

                first_relevant_rank = rank

                break

        if first_relevant_rank in distribution:

            distribution[
                first_relevant_rank
            ] += 1

        else:

            distribution[
                "not_retrieved"
            ] += 1

    return distribution


# ==================================================
# CLASSIFICATION METRICS
# ==================================================

def calculate_classification_metrics(
    evaluation_results
):
    """
    Calculate relevance classification metrics.

    Relevant query:
        accepted == True

    Irrelevant query:
        accepted == False
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
            accuracy
    }


# ==================================================
# SOURCE DISTRIBUTION
# ==================================================

def calculate_source_distribution(
    evaluation_results
):
    """
    Count how frequently each document appears
    in the retrieved top-k results.
    """

    distribution = {}

    for result in evaluation_results:

        for chunk in result[
            "results"
        ]:

            document = chunk.get(
                "document",
                "Unknown"
            )

            distribution[
                document
            ] = (
                distribution.get(
                    document,
                    0
                ) + 1
            )

    return distribution