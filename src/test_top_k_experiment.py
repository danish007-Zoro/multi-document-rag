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
    sys.path.insert(0, SRC_DIR)


# ==================================================
# IMPORTS
# ==================================================

from rag_pipeline import RAGPipeline

from multi_document_evaluation import (
    TEST_QUERIES,

    # Document-level metrics
    evaluate_document_attribution,
    calculate_document_hit_rate,
    calculate_document_hit_at_1,
    calculate_document_mrr,

    # Chunk-level metrics
    calculate_chunk_hit_rate,
    calculate_chunk_hit_at_1,
    calculate_chunk_mrr,

    # Classification
    calculate_classification_metrics,
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


# K VALUES TO EXPERIMENT WITH
TOP_K_VALUES = [
    1,
    3,
    5,
    10
]


DISTANCE_THRESHOLD = 1.4


# ==================================================
# HELPER
# ==================================================

def print_metric_row(
    k,
    document_hit_k,
    chunk_hit_k,
    document_mrr,
    chunk_mrr,
    classification
):
    """
    Print one row of the Top-K comparison table.
    """

    print(
        f"{k:<5}"
        f"{document_hit_k * 100:>12.2f}%"
        f"{chunk_hit_k * 100:>12.2f}%"
        f"{document_mrr:>12.4f}"
        f"{chunk_mrr:>12.4f}"
        f"{classification['precision'] * 100:>12.2f}%"
        f"{classification['recall'] * 100:>12.2f}%"
        f"{classification['f1_score'] * 100:>12.2f}%"
        f"{classification['false_positive']:>8}"
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 90)
    print("TOP-K RETRIEVAL EXPERIMENT")
    print("=" * 90)

    print(
        "\nThis experiment evaluates the same benchmark "
        "at multiple retrieval K values."
    )

    print(
        "\nK values:"
        f" {TOP_K_VALUES}"
    )

    print(
        f"Distance Threshold: "
        f"{DISTANCE_THRESHOLD}"
    )


    # ==================================================
    # VALIDATE DOCUMENTS
    # ==================================================

    print("\n")
    print("=" * 90)
    print("VALIDATING DOCUMENTS")
    print("=" * 90)

    for path in PDF_PATHS:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        print(
            f"✓ {os.path.basename(path)}"
        )


    # ==================================================
    # LOAD DOCUMENTS
    # ==================================================

    print("\n")
    print("=" * 90)
    print("LOADING DOCUMENTS")
    print("=" * 90)

    # Use the largest K once so the same
    # pipeline/index is used for every experiment.
    max_k = max(TOP_K_VALUES)

    pipeline = RAGPipeline(
        distance_threshold=DISTANCE_THRESHOLD,
        top_k=max_k
    )

    stats = pipeline.load_documents(
        PDF_PATHS
    )


    # ==================================================
    # DOCUMENT STATISTICS
    # ==================================================

    print("\n")
    print("=" * 90)
    print("DOCUMENT STATISTICS")
    print("=" * 90)

    for document in stats["documents"]:

        print(
            f"\n{document['name']}"
        )

        print(
            f"  Pages  : "
            f"{document['pages']}"
        )

        print(
            f"  Chunks : "
            f"{document['chunks']}"
        )

    print("\n")

    print(
        f"Total Documents : "
        f"{stats['total_documents']}"
    )

    print(
        f"Total Pages     : "
        f"{stats['total_pages']}"
    )

    print(
        f"Total Chunks    : "
        f"{stats['total_chunks']}"
    )

    print(
        f"Embedding Dim   : "
        f"{stats['embedding_dimension']}"
    )

    print(
        f"FAISS Vectors   : "
        f"{stats['vectors']}"
    )


    # ==================================================
    # EXPERIMENT
    # ==================================================

    all_results = {}


    for k in TOP_K_VALUES:

        print("\n")
        print("=" * 90)

        print(
            f"RUNNING EXPERIMENT: TOP-K = {k}"
        )

        print("=" * 90)

        print(
            f"\nQueries : "
            f"{len(TEST_QUERIES)}"
        )

        print(
            f"Top-K   : "
            f"{k}"
        )

        print(
            f"Threshold : "
            f"{DISTANCE_THRESHOLD}"
        )


        # --------------------------------------------------
        # Run evaluation
        # --------------------------------------------------

        evaluation_results = (
            evaluate_document_attribution(

                TEST_QUERIES,

                pipeline.model,

                pipeline.index,

                pipeline.chunks,

                top_k=k,

                distance_threshold=
                    DISTANCE_THRESHOLD
            )
        )


        # --------------------------------------------------
        # Document metrics
        # --------------------------------------------------

        document_hit_k = (
            calculate_document_hit_rate(
                evaluation_results,
                k=k
            )
        )

        document_hit_1 = (
            calculate_document_hit_at_1(
                evaluation_results
            )
        )

        document_mrr = (
            calculate_document_mrr(
                evaluation_results
            )
        )


        # --------------------------------------------------
        # Chunk metrics
        # --------------------------------------------------

        chunk_hit_k = (
            calculate_chunk_hit_rate(
                evaluation_results,
                k=k
            )
        )

        chunk_hit_1 = (
            calculate_chunk_hit_at_1(
                evaluation_results
            )
        )

        chunk_mrr = (
            calculate_chunk_mrr(
                evaluation_results
            )
        )


        # --------------------------------------------------
        # Classification metrics
        # --------------------------------------------------

        classification = (
            calculate_classification_metrics(
                evaluation_results
            )
        )


        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        all_results[k] = {

            "evaluation_results":
                evaluation_results,

            "document_hit_k":
                document_hit_k,

            "document_hit_1":
                document_hit_1,

            "document_mrr":
                document_mrr,

            "chunk_hit_k":
                chunk_hit_k,

            "chunk_hit_1":
                chunk_hit_1,

            "chunk_mrr":
                chunk_mrr,

            "classification":
                classification
        }


        # --------------------------------------------------
        # Print current K result
        # --------------------------------------------------

        print("\n")
        print("-" * 90)
        print(
            f"RESULTS FOR K = {k}"
        )
        print("-" * 90)

        print(
            f"\nDocument Hit@1   : "
            f"{document_hit_1 * 100:.2f}%"
        )

        print(
            f"Document Hit@{k}   : "
            f"{document_hit_k * 100:.2f}%"
        )

        print(
            f"Document MRR     : "
            f"{document_mrr:.4f}"
        )

        print(
            f"\nChunk Hit@1      : "
            f"{chunk_hit_1 * 100:.2f}%"
        )

        print(
            f"Chunk Hit@{k}      : "
            f"{chunk_hit_k * 100:.2f}%"
        )

        print(
            f"Chunk MRR        : "
            f"{chunk_mrr:.4f}"
        )

        print(
            f"\nPrecision        : "
            f"{classification['precision'] * 100:.2f}%"
        )

        print(
            f"Recall           : "
            f"{classification['recall'] * 100:.2f}%"
        )

        print(
            f"F1 Score         : "
            f"{classification['f1_score'] * 100:.2f}%"
        )

        print(
            f"Accuracy         : "
            f"{classification['accuracy'] * 100:.2f}%"
        )

        print(
            f"False Positives  : "
            f"{classification['false_positive']}"
        )

        print(
            f"False Negatives  : "
            f"{classification['false_negative']}"
        )


    # ==================================================
    # FINAL COMPARISON
    # ==================================================

    print("\n")
    print("=" * 90)
    print("TOP-K COMPARISON")
    print("=" * 90)

    print()

    print(
        f"{'K':<5}"
        f"{'Doc Hit@K':>12}"
        f"{'Chunk Hit@K':>12}"
        f"{'Doc MRR':>12}"
        f"{'Chunk MRR':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'FP':>8}"
    )

    print("-" * 90)


    for k in TOP_K_VALUES:

        result = all_results[k]

        print_metric_row(

            k,

            result["document_hit_k"],

            result["chunk_hit_k"],

            result["document_mrr"],

            result["chunk_mrr"],

            result["classification"]
        )


    # ==================================================
    # HIT@1 COMPARISON
    # ==================================================

    print("\n")
    print("=" * 90)
    print("HIT@1 COMPARISON")
    print("=" * 90)

    print()

    print(
        f"{'K':<5}"
        f"{'Doc Hit@1':>15}"
        f"{'Chunk Hit@1':>15}"
    )

    print("-" * 40)

    for k in TOP_K_VALUES:

        result = all_results[k]

        print(
            f"{k:<5}"
            f"{result['document_hit_1'] * 100:>14.2f}%"
            f"{result['chunk_hit_1'] * 100:>14.2f}%"
        )


    # ==================================================
    # BEST K ANALYSIS
    # ==================================================

    print("\n")
    print("=" * 90)
    print("BEST K ANALYSIS")
    print("=" * 90)


    # --------------------------------------------------
    # Best document Hit@K
    # --------------------------------------------------

    best_document_hit_k = max(

        TOP_K_VALUES,

        key=lambda k:
            all_results[k]["document_hit_k"]
    )

    print(
        f"\nBest Document Hit@K : "
        f"K = {best_document_hit_k}"
    )

    print(
        f"Score                : "
        f"{all_results[best_document_hit_k]['document_hit_k'] * 100:.2f}%"
    )


    # --------------------------------------------------
    # Best chunk Hit@K
    # --------------------------------------------------

    best_chunk_hit_k = max(

        TOP_K_VALUES,

        key=lambda k:
            all_results[k]["chunk_hit_k"]
    )

    print(
        f"\nBest Chunk Hit@K    : "
        f"K = {best_chunk_hit_k}"
    )

    print(
        f"Score                : "
        f"{all_results[best_chunk_hit_k]['chunk_hit_k'] * 100:.2f}%"
    )


    # --------------------------------------------------
    # Best document MRR
    # --------------------------------------------------

    best_document_mrr = max(

        TOP_K_VALUES,

        key=lambda k:
            all_results[k]["document_mrr"]
    )

    print(
        f"\nBest Document MRR   : "
        f"K = {best_document_mrr}"
    )

    print(
        f"Score                : "
        f"{all_results[best_document_mrr]['document_mrr']:.4f}"
    )


    # --------------------------------------------------
    # Best chunk MRR
    # --------------------------------------------------

    best_chunk_mrr = max(

        TOP_K_VALUES,

        key=lambda k:
            all_results[k]["chunk_mrr"]
    )

    print(
        f"\nBest Chunk MRR      : "
        f"K = {best_chunk_mrr}"
    )

    print(
        f"Score                : "
        f"{all_results[best_chunk_mrr]['chunk_mrr']:.4f}"
    )


    # --------------------------------------------------
    # Best F1
    # --------------------------------------------------

    best_f1 = max(

        TOP_K_VALUES,

        key=lambda k:
            all_results[k]["classification"]["f1_score"]
    )

    print(
        f"\nBest Classification F1 : "
        f"K = {best_f1}"
    )

    print(
        f"Score                  : "
        f"{all_results[best_f1]['classification']['f1_score'] * 100:.2f}%"
    )


    # ==================================================
    # FALSE POSITIVE ANALYSIS
    # ==================================================

    print("\n")
    print("=" * 90)
    print("FALSE POSITIVE ANALYSIS")
    print("=" * 90)

    print()

    for k in TOP_K_VALUES:

        classification = (
            all_results[k]["classification"]
        )

        print(
            f"K = {k:<3}"
            f" | False Positives: "
            f"{classification['false_positive']}"
            f" | False Negatives: "
            f"{classification['false_negative']}"
        )


    # ==================================================
    # INTERPRETATION
    # ==================================================

    print("\n")
    print("=" * 90)
    print("EXPERIMENT INTERPRETATION")
    print("=" * 90)

    print(
        "\nUse the comparison above to determine the "
        "best retrieval K."
    )

    print(
        "\nImportant:"
    )

    print(
        "  • Higher Hit@K generally means better recall."
    )

    print(
        "  • Higher MRR means relevant evidence appears "
        "higher in the ranking."
    )

    print(
        "  • False positives indicate irrelevant queries "
        "being incorrectly accepted."
    )

    print(
        "  • Increasing K is not automatically better."
    )

    print(
        "  • A larger K can retrieve more evidence but "
        "may also introduce irrelevant context."
    )

    print(
        "\nFor the final RAG configuration, consider "
        "retrieval quality together with generation "
        "and groundedness results."
    )


    # ==================================================
    # COMPLETE
    # ==================================================

    print("\n")
    print("=" * 90)
    print(
        "TOP-K RETRIEVAL EXPERIMENT COMPLETE"
    )
    print("=" * 90)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()