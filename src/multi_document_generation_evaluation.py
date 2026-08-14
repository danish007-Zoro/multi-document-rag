# ==================================================
# MULTI-DOCUMENT GENERATION QUALITY EVALUATION
# ==================================================


# ==================================================
# TEST QUERIES
# ==================================================

TEST_QUERIES = [

    # ------------------------------------------------
    # ARTIFICIAL INTELLIGENCE
    # ------------------------------------------------

    {
        "query": "What is Artificial Intelligence?",
        "relevant": True,

        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf",
            "Sample3.pdf"
        ],

        "expected_facts": [
            "think",
            "solve",
            "complex problems",
            "intelligence"
        ]
    },

    {
        "query": "What is AI?",
        "relevant": True,

        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf",
            "Sample3.pdf"
        ],

        "expected_facts": [
            "think",
            "solve",
            "complex problems"
        ]
    },


    # ------------------------------------------------
    # AI HISTORY
    # ------------------------------------------------

    {
        "query": "When was the Turing test invented?",
        "relevant": True,

        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf",
            "Sample3.pdf"
        ],

        "expected_facts": [
            "1950",
            "Alan Turing"
        ]
    },

    {
        "query": "When was ELIZA created?",
        "relevant": True,

        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf",
            "Sample3.pdf"
        ],

        "expected_facts": [
            "1960"
        ]
    },

    {
        "query": "When was Siri announced?",
        "relevant": True,

        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf"
        ],

        "expected_facts": [
            "2011",
            "Siri",
            "Apple"
        ]
    },

    {
        "query": "When was OpenAI founded?",
        "relevant": True,

        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf"
        ],

        "expected_facts": [
            "2015",
            "OpenAI"
        ]
    },

    {
        "query": "What computer defeated a world chess champion?",
        "relevant": True,

        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf"
        ],

        "expected_facts": [
            "IBM",
            "Deep Blue",
            "chess"
        ]
    },


    # ------------------------------------------------
    # MACHINE LEARNING
    # ------------------------------------------------

    {
        "query": "What is Machine Learning?",
        "relevant": True,

        "expected_documents": [
            "Sample3.pdf"
        ],

        "expected_facts": [
            "learn",
            "data",
            "algorithms"
        ]
    },

    {
        "query": "How does machine learning work?",
        "relevant": True,

        "expected_documents": [
            "Sample3.pdf"
        ],

        "expected_facts": [
            "data",
            "train",
            "algorithm",
            "prediction"
        ]
    },

    {
        "query": "What is supervised learning?",
        "relevant": True,

        "expected_documents": [
            "Sample3.pdf"
        ],

        "expected_facts": [
            "training",
            "examples",
            "data"
        ]
    },

    {
        "query": "What is unsupervised learning?",
        "relevant": True,

        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf",
            "Sample3.pdf"
        ],

        "expected_facts": [
            "unlabelled data",
            "patterns",
            "conclusions"
        ]
    },

    {
        "query": "What are the applications of machine learning?",
        "relevant": True,

        "expected_documents": [
            "Sample3.pdf"
        ],

        "expected_facts": [
            "fraud detection",
            "banking",
            "advertising"
        ]
    },

    {
        "query": "How is machine learning used in fraud detection?",
        "relevant": True,

        "expected_documents": [
            "Sample3.pdf"
        ],

        "expected_facts": [
            "patterns",
            "transaction",
            "fraudulent"
        ]
    },

    {
        "query": "How is machine learning used in gaming?",
        "relevant": True,

        "expected_documents": [
            "Sample3.pdf"
        ],

        "expected_facts": [
            "gaming",
            "balanced gameplay",
            "game developers"
        ]
    },

    {
        "query": "What are the challenges of machine learning?",
        "relevant": True,

        "expected_documents": [
            "Sample3.pdf"
        ],

        "expected_facts": [
            "data",
            "bias",
            "privacy"
        ]
    },


    # ------------------------------------------------
    # CROSS-DOCUMENT
    # ------------------------------------------------

    {
        "query": "How are Artificial Intelligence and Machine Learning related?",
        "relevant": True,

        "expected_documents": [
            "Sample1.pdf",
            "Sample2.pdf",
            "Sample3.pdf"
        ],

        "expected_facts": [
            "AI",
            "machine learning",
            "subfield",
            "algorithms"
        ]
    },


    # ------------------------------------------------
    # IRRELEVANT
    # ------------------------------------------------

    {
        "query": "What is the capital of France?",
        "relevant": False,

        "expected_documents": [],

        "expected_facts": []
    },

    {
        "query": "Who is the Prime Minister of India?",
        "relevant": False,

        "expected_documents": [],

        "expected_facts": []
    },

    {
        "query": "How do you bake a chocolate cake?",
        "relevant": False,

        "expected_documents": [],

        "expected_facts": []
    },

    {
        "query": "What is the boiling point of water?",
        "relevant": False,

        "expected_documents": [],

        "expected_facts": []
    },

    {
        "query": "Who wrote Romeo and Juliet?",
        "relevant": False,

        "expected_documents": [],

        "expected_facts": []
    },

    {
        "query": "What is the tallest mountain in the world?",
        "relevant": False,

        "expected_documents": [],

        "expected_facts": []
    }
]


# ==================================================
# REFUSAL TEXT
# ==================================================

REFUSAL = (
    "I could not find the answer "
    "in the provided documents."
)


# ==================================================
# NORMALIZE TEXT
# ==================================================

def normalize_text(text):
    """
    Normalize text for simple factual matching.
    """

    return (
        text
        .lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("-", " ")
    )


# ==================================================
# FACT EVALUATION
# ==================================================

def evaluate_facts(
    answer,
    expected_facts
):
    """
    Check how many expected facts appear
    in the generated answer.

    Returns:
        dict containing matched/missing facts
        and factual coverage.
    """

    normalized_answer = normalize_text(
        answer
    )

    matched = []
    missing = []

    for fact in expected_facts:

        normalized_fact = normalize_text(
            fact
        )

        if normalized_fact in normalized_answer:

            matched.append(
                fact
            )

        else:

            missing.append(
                fact
            )

    total = len(
        expected_facts
    )

    coverage = (
        len(matched) / total
        if total > 0
        else 1.0
    )

    return {
        "matched_facts": matched,
        "missing_facts": missing,
        "fact_coverage": coverage
    }


# ==================================================
# SOURCE EVALUATION
# ==================================================

def evaluate_sources(
    results,
    expected_documents
):
    """
    Check whether retrieved documents contain
    at least one expected document.

    Returns:
        source hit information.
    """

    retrieved_documents = []

    for chunk in results:

        document = chunk.get(
            "document",
            "Unknown"
        )

        if document not in retrieved_documents:

            retrieved_documents.append(
                document
            )

    if not expected_documents:

        source_hit = None

    else:

        source_hit = any(
            document in expected_documents
            for document in retrieved_documents
        )

    return {
        "retrieved_documents":
            retrieved_documents,

        "source_hit":
            source_hit
    }


# ==================================================
# GENERATION EVALUATION
# ==================================================

def evaluate_generation(
    test_queries,
    pipeline,
    top_k=3,
    distance_threshold=1.4
):
    """
    Evaluate factual quality of generated answers.

    Evaluation includes:

        1. Retrieval acceptance
        2. Expected fact coverage
        3. Source attribution
        4. Correct refusal for irrelevant queries

    Returns:
        list of evaluation results.
    """

    evaluation_results = []

    for item in test_queries:

        query = item["query"]

        expected_relevant = (
            item["relevant"]
        )

        expected_facts = (
            item.get(
                "expected_facts",
                []
            )
        )

        expected_documents = (
            item.get(
                "expected_documents",
                []
            )
        )


        # ------------------------------------------------
        # Run RAG pipeline
        # ------------------------------------------------

        result = pipeline.ask(
            query
        )

        accepted = (
            result["accepted"]
        )

        answer = (
            result["answer"]
        )

        best_distance = (
            result["best_distance"]
        )

        results = (
            result["results"]
        )


        # ------------------------------------------------
        # Fact evaluation
        # ------------------------------------------------

        fact_result = evaluate_facts(
            answer,
            expected_facts
        )


        # ------------------------------------------------
        # Source evaluation
        # ------------------------------------------------

        source_result = evaluate_sources(
            results,
            expected_documents
        )


        # ------------------------------------------------
        # Determine pass/fail
        # ------------------------------------------------

        if expected_relevant:

            # Relevant query must:
            # 1. Be accepted
            # 2. Contain at least 50%
            #    of expected facts
            # 3. Retrieve an expected document

            passed = (
                accepted
                and
                fact_result[
                    "fact_coverage"
                ] >= 0.50
                and
                source_result[
                    "source_hit"
                ]
            )

        else:

            # Irrelevant query must be rejected
            # and produce the exact refusal.

            passed = (
                not accepted
                and
                answer.strip()
                == REFUSAL
            )


        evaluation_results.append({

            "query":
                query,

            "expected_relevant":
                expected_relevant,

            "expected_documents":
                expected_documents,

            "expected_facts":
                expected_facts,

            "accepted":
                accepted,

            "best_distance":
                best_distance,

            "answer":
                answer,

            "results":
                results,

            "retrieved_documents":
                source_result[
                    "retrieved_documents"
                ],

            "source_hit":
                source_result[
                    "source_hit"
                ],

            "matched_facts":
                fact_result[
                    "matched_facts"
                ],

            "missing_facts":
                fact_result[
                    "missing_facts"
                ],

            "fact_coverage":
                fact_result[
                    "fact_coverage"
                ],

            "passed":
                passed
        })


    return evaluation_results


# ==================================================
# GENERATION METRICS
# ==================================================

def calculate_generation_metrics(
    evaluation_results
):
    """
    Calculate overall generation metrics.
    """

    total = len(
        evaluation_results
    )

    passed = sum(
        1
        for result in evaluation_results
        if result["passed"]
    )

    failed = (
        total - passed
    )

    pass_rate = (
        passed / total
        if total > 0
        else 0.0
    )


    relevant_results = [
        result
        for result in evaluation_results
        if result["expected_relevant"]
    ]

    irrelevant_results = [
        result
        for result in evaluation_results
        if not result["expected_relevant"]
    ]


    # Average fact coverage
    average_fact_coverage = (
        sum(
            result["fact_coverage"]
            for result in relevant_results
        )
        /
        len(relevant_results)
        if relevant_results
        else 0.0
    )


    # Relevant answer pass rate
    relevant_pass_rate = (
        sum(
            1
            for result in relevant_results
            if result["passed"]
        )
        /
        len(relevant_results)
        if relevant_results
        else 0.0
    )


    # Refusal accuracy
    refusal_accuracy = (
        sum(
            1
            for result in irrelevant_results
            if result["passed"]
        )
        /
        len(irrelevant_results)
        if irrelevant_results
        else 0.0
    )


    return {

        "total":
            total,

        "passed":
            passed,

        "failed":
            failed,

        "pass_rate":
            pass_rate,

        "average_fact_coverage":
            average_fact_coverage,

        "relevant_pass_rate":
            relevant_pass_rate,

        "refusal_accuracy":
            refusal_accuracy
    }