import os
import sys
import numpy as np

from sentence_transformers import CrossEncoder

from rag_pipeline import RAGPipeline
from embeddings import load_embedding_model


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
    sys.path.insert(0, SRC_DIR)


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

# Best threshold obtained from the previous
# claim-level embedding experiment.
EMBEDDING_GROUNDEDNESS_THRESHOLD = 0.75

NLI_MODEL_NAME = (
    "cross-encoder/nli-deberta-v3-base"
)


# ==================================================
# BENCHMARK
# ==================================================
#
# These are the SAME 20 claims used in the
# previous claim-level groundedness evaluation.
#
# Each claim is paired with the query that should
# retrieve its supporting evidence.
#
# expected_grounded:
#     True  -> grounded
#     False -> ungrounded
#
# ==================================================

CLAIMS = [

    {
        "query":
            "When was the Turing test invented?",
        "claim":
            "The Turing test was invented in 1950 by Alan Turing.",
        "expected_grounded":
            True
    },

    {
        "query":
            "When was Siri announced?",
        "claim":
            "Siri was announced as a digital assistant by Apple in 2011.",
        "expected_grounded":
            True
    },

    {
        "query":
            "What is Machine Learning?",
        "claim":
            "Machine learning gives computers the ability to learn without being explicitly programmed.",
        "expected_grounded":
            True
    },

    {
        "query":
            "What is Artificial Intelligence?",
        "claim":
            "Artificial Intelligence helps machines solve complex problems using humanlike intelligence.",
        "expected_grounded":
            True
    },

    {
        "query":
            "How is machine learning used in gaming?",
        "claim":
            "Machine learning can be used to create balanced gameplay.",
        "expected_grounded":
            True
    },

    {
        "query":
            "How is machine learning used in gaming?",
        "claim":
            "Machine learning is used in gaming to make games more interesting.",
        "expected_grounded":
            True
    },

    {
        "query":
            "When was OpenAI founded?",
        "claim":
            "OpenAI was founded in 2015.",
        "expected_grounded":
            True
    },

    {
        "query":
            "What are the applications of machine learning?",
        "claim":
            "Machine learning is used for fraud detection.",
        "expected_grounded":
            True
    },

    {
        "query":
            "What is Machine Learning?",
        "claim":
            "Machine learning can be used to predict outcomes.",
        "expected_grounded":
            True
    },

    {
        "query":
            "What is Machine Learning?",
        "claim":
            "Machine learning can classify information.",
        "expected_grounded":
            True
    },

    {
        "query":
            "When was Siri announced?",
        "claim":
            "Siri was announced by Apple in 2008.",
        "expected_grounded":
            False
    },

    {
        "query":
            "When was OpenAI founded?",
        "claim":
            "OpenAI was founded by Bill Gates in 2015.",
        "expected_grounded":
            False
    },

    {
        "query":
            "When was OpenAI founded?",
        "claim":
            "OpenAI was founded in 2012.",
        "expected_grounded":
            False
    },

    {
        "query":
            "When was the Turing test invented?",
        "claim":
            "The Turing test was invented by Albert Einstein.",
        "expected_grounded":
            False
    },

    {
        "query":
            "What is Machine Learning?",
        "claim":
            "Machine learning completely replaces human decision-making.",
        "expected_grounded":
            False
    },

    {
        "query":
            "What is Machine Learning?",
        "claim":
            "Machine learning predicts outcomes with perfect accuracy.",
        "expected_grounded":
            False
    },

    {
        "query":
            "How is machine learning used in gaming?",
        "claim":
            "Machine learning is only used in gaming.",
        "expected_grounded":
            False
    },

    {
        "query":
            "When was Siri announced?",
        "claim":
            "Siri was created by Google.",
        "expected_grounded":
            False
    },

    {
        "query":
            "When was the Turing test invented?",
        "claim":
            "Alan Turing invented the Turing test at Oxford University.",
        "expected_grounded":
            False
    },

    {
        "query":
            "What is Machine Learning?",
        "claim":
            "Machine learning can guarantee that predictions will always be correct.",
        "expected_grounded":
            False
    }
]


# ==================================================
# SOFTMAX
# ==================================================

def softmax(values):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    values = (
        values
        -
        np.max(values)
    )

    probabilities = np.exp(
        values
    )

    probabilities /= np.sum(
        probabilities
    )

    return probabilities


# ==================================================
# LOAD NLI MODEL
# ==================================================

def load_nli_model():

    print(
        f"Loading NLI model: "
        f"{NLI_MODEL_NAME}"
    )

    return CrossEncoder(
        NLI_MODEL_NAME
    )


# ==================================================
# NLI PREDICTION
# ==================================================

def predict_nli(
    nli_model,
    premise,
    hypothesis
):

    scores = nli_model.predict(
        [
            (
                premise,
                hypothesis
            )
        ]
    )

    scores = np.asarray(
        scores
    )


    # ----------------------------------------------
    # CrossEncoder normally returns one row
    # containing three logits.
    # ----------------------------------------------

    if scores.ndim == 2:

        scores = scores[0]


    probabilities = softmax(
        scores
    )


    # cross-encoder/nli-deberta-v3-base
    #
    # Label order:
    #   0 = contradiction
    #   1 = entailment
    #   2 = neutral
    #
    contradiction_score = float(
        probabilities[0]
    )

    entailment_score = float(
        probabilities[1]
    )

    neutral_score = float(
        probabilities[2]
    )


    scores_by_label = {

        "contradiction":
            contradiction_score,

        "entailment":
            entailment_score,

        "neutral":
            neutral_score
    }


    label = max(
        scores_by_label,
        key=scores_by_label.get
    )


    return {

        "label": label,

        "confidence":
            scores_by_label[label],

        "contradiction":
            contradiction_score,

        "entailment":
            entailment_score,

        "neutral":
            neutral_score
    }


# ==================================================
# EMBEDDING SIMILARITY
# ==================================================

def calculate_embedding_similarity(
    embedding_model,
    claim,
    evidence
):

    embeddings = embedding_model.encode(
        [
            claim,
            evidence
        ],
        normalize_embeddings=True
    )


    claim_embedding = embeddings[0]

    evidence_embedding = embeddings[1]


    similarity = float(
        np.dot(
            claim_embedding,
            evidence_embedding
        )
    )


    return similarity


# ==================================================
# FIND BEST EVIDENCE
# ==================================================

def get_evidence_text(
    result
):

    text = result.get(
        "text"
    )

    if text:

        return text


    return ""


# ==================================================
# EMBEDDING METHOD
# ==================================================

def evaluate_embedding_method(
    embedding_model,
    retrieval_results
):

    best_similarity = -1.0

    best_evidence = None


    for result in retrieval_results:

        evidence = get_evidence_text(
            result
        )

        if not evidence:
            continue


        similarity = (
            calculate_embedding_similarity(
                embedding_model,
                CURRENT_CLAIM,
                evidence
            )
        )


        if similarity > best_similarity:

            best_similarity = similarity

            best_evidence = result


    grounded = (
        best_similarity
        >=
        EMBEDDING_GROUNDEDNESS_THRESHOLD
    )


    return {

        "grounded":
            grounded,

        "score":
            best_similarity,

        "evidence":
            best_evidence
    }


# ==================================================
# NLI METHOD
# ==================================================

def evaluate_nli_method(
    nli_model,
    retrieval_results
):

    best_entailment = -1.0

    best_result = None

    best_nli = None


    for result in retrieval_results:

        evidence = get_evidence_text(
            result
        )

        if not evidence:
            continue


        nli_result = predict_nli(

            nli_model,

            evidence,

            CURRENT_CLAIM
        )


        entailment_score = (
            nli_result["entailment"]
        )


        if entailment_score > best_entailment:

            best_entailment = (
                entailment_score
            )

            best_result = result

            best_nli = nli_result


    grounded = (
        best_nli is not None
        and
        best_nli["label"] == "entailment"
    )


    return {

        "grounded":
            grounded,

        "score":
            best_entailment,

        "label":
            (
                best_nli["label"]
                if best_nli
                else "unknown"
            ),

        "confidence":
            (
                best_nli["confidence"]
                if best_nli
                else 0.0
            ),

        "evidence":
            best_result
    }


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(
    results,
    method_key
):

    true_positive = 0

    true_negative = 0

    false_positive = 0

    false_negative = 0


    for result in results:

        actual = (
            result["expected_grounded"]
        )

        predicted = (
            result[method_key]
        )


        if actual and predicted:

            true_positive += 1

        elif not actual and not predicted:

            true_negative += 1

        elif not actual and predicted:

            false_positive += 1

        elif actual and not predicted:

            false_negative += 1


    total = len(results)


    precision_denominator = (
        true_positive
        +
        false_positive
    )

    recall_denominator = (
        true_positive
        +
        false_negative
    )

    accuracy_denominator = total


    if precision_denominator > 0:

        precision = (
            true_positive
            /
            precision_denominator
        )

    else:

        precision = 0.0


    if recall_denominator > 0:

        recall = (
            true_positive
            /
            recall_denominator
        )

    else:

        recall = 0.0


    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            *
            precision
            *
            recall
            /
            (
                precision
                +
                recall
            )
        )

    else:

        f1 = 0.0


    accuracy = (
        (
            true_positive
            +
            true_negative
        )
        /
        accuracy_denominator
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
# PRINT METRICS
# ==================================================

def print_metrics(
    title,
    metrics
):

    print("\n")

    print(
        title
    )

    print(
        "-" * 70
    )

    print(
        f"True Positives      : "
        f"{metrics['true_positive']}"
    )

    print(
        f"True Negatives      : "
        f"{metrics['true_negative']}"
    )

    print(
        f"False Positives     : "
        f"{metrics['false_positive']}"
    )

    print(
        f"False Negatives     : "
        f"{metrics['false_negative']}"
    )

    print(
        f"Precision           : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall              : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score            : "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print(
        f"Accuracy            : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )


# ==================================================
# MAIN
# ==================================================

def main():

    global CURRENT_CLAIM


    print("=" * 90)

    print(
        "EMBEDDING VS NLI GROUNDEDNESS "
        "COMPARISON"
    )

    print("=" * 90)


    # ==================================================
    # VALIDATE DOCUMENTS
    # ==================================================

    print("\n")

    print(
        "Validating documents..."
    )


    for path in PDF_PATHS:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        print(
            f"✓ {os.path.basename(path)}"
        )


    # ==================================================
    # LOAD PIPELINE
    # ==================================================

    print("\n")

    print(
        "Loading documents..."
    )


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

    print("=" * 90)

    print(
        "DOCUMENT STATISTICS"
    )

    print("=" * 90)


    print(
        f"Documents : "
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
    # LOAD MODELS
    # ==================================================

    print("\n")

    print(
        "Loading embedding model..."
    )


    embedding_model = (
        load_embedding_model()
    )


    print("\n")

    print(
        "Loading NLI model..."
    )


    nli_model = load_nli_model()


    # ==================================================
    # EVALUATION
    # ==================================================

    print("\n")

    print("=" * 90)

    print(
        "RUNNING COMPARISON"
    )

    print("=" * 90)


    results = []


    for number, item in enumerate(
        CLAIMS,
        start=1
    ):

        CURRENT_CLAIM = (
            item["claim"]
        )


        print("\n")

        print(
            "-" * 90
        )

        print(
            f"[{number:02d}] "
            f"{item['claim']}"
        )

        print(
            f"Query: "
            f"{item['query']}"
        )

        print(
            f"Expected: "
            f"{'GROUNDED' if item['expected_grounded'] else 'UNGROUNDED'}"
        )


        # ----------------------------------------------
        # Retrieve evidence using the same RAG
        # retrieval configuration.
        # ----------------------------------------------

        retrieval = pipeline.ask(
            item["query"]
        )


        retrieval_results = (
            retrieval.get(
                "results",
                []
            )
        )


        print(
            f"Retrieval Accepted: "
            f"{retrieval.get('accepted', False)}"
        )

        print(
            f"Best Distance     : "
            f"{retrieval.get('best_distance', float('inf')):.4f}"
        )


        # ----------------------------------------------
        # Embedding evaluation
        # ----------------------------------------------

        embedding_result = (
            evaluate_embedding_method(
                embedding_model,
                retrieval_results
            )
        )


        # ----------------------------------------------
        # NLI evaluation
        # ----------------------------------------------

        nli_result = (
            evaluate_nli_method(
                nli_model,
                retrieval_results
            )
        )


        embedding_prediction = (
            embedding_result["grounded"]
        )

        nli_prediction = (
            nli_result["grounded"]
        )


        embedding_correct = (
            embedding_prediction
            ==
            item["expected_grounded"]
        )


        nli_correct = (
            nli_prediction
            ==
            item["expected_grounded"]
        )


        # ----------------------------------------------
        # Store result
        # ----------------------------------------------

        results.append({

            "number":
                number,

            "claim":
                item["claim"],

            "expected_grounded":
                item["expected_grounded"],

            "embedding_prediction":
                embedding_prediction,

            "embedding_score":
                embedding_result["score"],

            "nli_prediction":
                nli_prediction,

            "nli_label":
                nli_result["label"],

            "nli_score":
                nli_result["score"],

            "embedding_correct":
                embedding_correct,

            "nli_correct":
                nli_correct
        })


        # ----------------------------------------------
        # Display
        # ----------------------------------------------

        print("\n")

        print(
            "Embedding Method"
        )

        print(
            f"  Prediction : "
            f"{'GROUNDED' if embedding_prediction else 'UNGROUNDED'}"
        )

        print(
            f"  Similarity : "
            f"{embedding_result['score']:.4f}"
        )

        print(
            f"  Correct    : "
            f"{'YES' if embedding_correct else 'NO'}"
        )


        print("\n")

        print(
            "NLI Method"
        )

        print(
            f"  Prediction : "
            f"{'GROUNDED' if nli_prediction else 'UNGROUNDED'}"
        )

        print(
            f"  Label      : "
            f"{nli_result['label']}"
        )

        print(
            f"  Score      : "
            f"{nli_result['score']:.4f}"
        )

        print(
            f"  Correct    : "
            f"{'YES' if nli_correct else 'NO'}"
        )


        # ----------------------------------------------
        # Disagreement
        # ----------------------------------------------

        if (
            embedding_prediction
            !=
            nli_prediction
        ):

            print("\n")

            print(
                ">>> METHOD DISAGREEMENT <<<"
            )


    # ==================================================
    # CALCULATE METRICS
    # ==================================================

    embedding_metrics = (
        calculate_metrics(
            results,
            "embedding_prediction"
        )
    )


    nli_metrics = (
        calculate_metrics(
            results,
            "nli_prediction"
        )
    )


    # ==================================================
    # SUMMARY
    # ==================================================

    print("\n\n")

    print("=" * 90)

    print(
        "GROUNDEDNESS COMPARISON SUMMARY"
    )

    print("=" * 90)


    print(
        f"\nTotal Claims : "
        f"{len(results)}"
    )


    print_metrics(
        "EMBEDDING SIMILARITY",
        embedding_metrics
    )


    print_metrics(
        "NLI ENTAILMENT",
        nli_metrics
    )


    # ==================================================
    # COMPARISON TABLE
    # ==================================================

    print("\n\n")

    print("=" * 90)

    print(
        "METHOD COMPARISON"
    )

    print("=" * 90)


    print()

    print(
        f"{'Metric':<20}"
        f"{'Embedding':>20}"
        f"{'NLI':>20}"
    )

    print(
        "-" * 60
    )


    print(
        f"{'Precision':<20}"
        f"{embedding_metrics['precision'] * 100:>19.2f}%"
        f"{nli_metrics['precision'] * 100:>19.2f}%"
    )

    print(
        f"{'Recall':<20}"
        f"{embedding_metrics['recall'] * 100:>19.2f}%"
        f"{nli_metrics['recall'] * 100:>19.2f}%"
    )

    print(
        f"{'F1 Score':<20}"
        f"{embedding_metrics['f1'] * 100:>19.2f}%"
        f"{nli_metrics['f1'] * 100:>19.2f}%"
    )

    print(
        f"{'Accuracy':<20}"
        f"{embedding_metrics['accuracy'] * 100:>19.2f}%"
        f"{nli_metrics['accuracy'] * 100:>19.2f}%"
    )

    print(
        f"{'False Positives':<20}"
        f"{embedding_metrics['false_positive']:>20}"
        f"{nli_metrics['false_positive']:>20}"
    )

    print(
        f"{'False Negatives':<20}"
        f"{embedding_metrics['false_negative']:>20}"
        f"{nli_metrics['false_negative']:>20}"
    )


    # ==================================================
    # DISAGREEMENT ANALYSIS
    # ==================================================

    disagreements = [

        result

        for result in results

        if (
            result["embedding_prediction"]
            !=
            result["nli_prediction"]
        )
    ]


    print("\n\n")

    print("=" * 90)

    print(
        "METHOD DISAGREEMENT ANALYSIS"
    )

    print("=" * 90)


    print(
        f"\nTotal disagreements : "
        f"{len(disagreements)}"
    )


    if not disagreements:

        print(
            "\nNo disagreements between "
            "the two methods."
        )

    else:

        for result in disagreements:

            print("\n")

            print(
                f"[{result['number']:02d}] "
                f"{result['claim']}"
            )

            print(
                f"Expected    : "
                f"{'GROUNDED' if result['expected_grounded'] else 'UNGROUNDED'}"
            )

            print(
                f"Embedding   : "
                f"{'GROUNDED' if result['embedding_prediction'] else 'UNGROUNDED'}"
            )

            print(
                f"Similarity  : "
                f"{result['embedding_score']:.4f}"
            )

            print(
                f"NLI         : "
                f"{'GROUNDED' if result['nli_prediction'] else 'UNGROUNDED'}"
            )

            print(
                f"NLI Label   : "
                f"{result['nli_label']}"
            )

            print(
                f"NLI Score   : "
                f"{result['nli_score']:.4f}"
            )


    # ==================================================
    # FINAL DECISION
    # ==================================================

    print("\n\n")

    print("=" * 90)

    print(
        "FINAL METHOD COMPARISON"
    )

    print("=" * 90)


    if (
        nli_metrics["f1"]
        >
        embedding_metrics["f1"]
    ):

        winner = "NLI Entailment"

    elif (
        embedding_metrics["f1"]
        >
        nli_metrics["f1"]
    ):

        winner = "Embedding Similarity"

    else:

        if (
            nli_metrics["accuracy"]
            >
            embedding_metrics["accuracy"]
        ):

            winner = "NLI Entailment"

        elif (
            embedding_metrics["accuracy"]
            >
            nli_metrics["accuracy"]
        ):

            winner = "Embedding Similarity"

        else:

            winner = "Tie"


    print(
        f"\nBest Method : "
        f"{winner}"
    )


    print(
        "\nEmbedding threshold used : "
        f"{EMBEDDING_GROUNDEDNESS_THRESHOLD}"
    )

    print(
        "NLI decision rule          : "
        "entailment label"
    )


    # ==================================================
    # COMPLETE
    # ==================================================

    print("\n")

    print("=" * 90)

    print(
        "GROUNDEDNESS COMPARISON COMPLETE"
    )

    print("=" * 90)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    main()