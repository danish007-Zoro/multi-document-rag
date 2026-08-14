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

    # Document-level evaluation
    evaluate_document_attribution,
    calculate_document_hit_rate,
    calculate_document_hit_at_1,
    calculate_document_mrr,
    calculate_document_rank_distribution,

    # Chunk-level evaluation
    calculate_chunk_hit_rate,
    calculate_chunk_hit_at_1,
    calculate_chunk_mrr,
    calculate_chunk_rank_distribution,

    # Classification
    calculate_classification_metrics,

    # Source distribution
    calculate_source_distribution
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

TOP_K = 3

DISTANCE_THRESHOLD = 1.4


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)
    print("MULTI-DOCUMENT RAG EVALUATION")
    print("=" * 70)


    # ==================================================
    # VALIDATE DOCUMENTS
    # ==================================================

    print("\nValidating documents...")

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

    print("\nLoading documents...")

    pipeline = RAGPipeline(
        distance_threshold=DISTANCE_THRESHOLD,
        top_k=TOP_K
    )

    stats = pipeline.load_documents(
        PDF_PATHS
    )


    # ==================================================
    # DOCUMENT STATISTICS
    # ==================================================

    print("\n")
    print("=" * 70)
    print("DOCUMENT STATISTICS")
    print("=" * 70)

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
    # EVALUATION CONFIGURATION
    # ==================================================

    print("\n")
    print("=" * 70)
    print("EVALUATION CONFIGURATION")
    print("=" * 70)

    print(
        f"Total Queries : "
        f"{len(TEST_QUERIES)}"
    )

    print(
        f"Top-K         : "
        f"{TOP_K}"
    )

    print(
        f"Threshold     : "
        f"{DISTANCE_THRESHOLD}"
    )


    # ==================================================
    # RUN EVALUATION
    # ==================================================

    print("\nRunning evaluation...")

    evaluation_results = (
        evaluate_document_attribution(
            TEST_QUERIES,
            pipeline.model,
            pipeline.index,
            pipeline.chunks,
            top_k=TOP_K,
            distance_threshold=DISTANCE_THRESHOLD
        )
    )


    # ==================================================
    # CALCULATE METRICS
    # ==================================================

    # ----------------------------------------------
    # Classification metrics
    # ----------------------------------------------

    classification = (
        calculate_classification_metrics(
            evaluation_results
        )
    )


    # ----------------------------------------------
    # Document Hit@K
    # ----------------------------------------------

    document_hit_at_k = (
        calculate_document_hit_rate(
            evaluation_results,
            k=TOP_K
        )
    )


    # ----------------------------------------------
    # Document Hit@1
    # ----------------------------------------------

    document_hit_at_1 = (
        calculate_document_hit_at_1(
            evaluation_results
        )
    )


    # ----------------------------------------------
    # Document MRR
    # ----------------------------------------------

    document_mrr = calculate_document_mrr(
        evaluation_results
    )


    # ----------------------------------------------
    # Document rank distribution
    # ----------------------------------------------

    document_rank_distribution = (
        calculate_document_rank_distribution(
            evaluation_results
        )
    )


    # ----------------------------------------------
    # Chunk Hit@K
    # ----------------------------------------------

    chunk_hit_at_k = (
        calculate_chunk_hit_rate(
            evaluation_results,
            k=TOP_K
        )
    )


    # ----------------------------------------------
    # Chunk Hit@1
    # ----------------------------------------------

    chunk_hit_at_1 = (
        calculate_chunk_hit_at_1(
            evaluation_results
        )
    )


    # ----------------------------------------------
    # Chunk MRR
    # ----------------------------------------------

    chunk_mrr = calculate_chunk_mrr(
        evaluation_results
    )


    # ----------------------------------------------
    # Chunk rank distribution
    # ----------------------------------------------

    chunk_rank_distribution = (
        calculate_chunk_rank_distribution(
            evaluation_results
        )
    )


    # ----------------------------------------------
    # Source distribution
    # ----------------------------------------------

    source_distribution = (
        calculate_source_distribution(
            evaluation_results
        )
    )


    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print("\n")
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)


    # ==================================================
    # CLASSIFICATION
    # ==================================================

    print("\n")
    print("CLASSIFICATION")
    print("-" * 70)

    print(
        f"True Positives      : "
        f"{classification['true_positive']}"
    )

    print(
        f"True Negatives      : "
        f"{classification['true_negative']}"
    )

    print(
        f"False Positives     : "
        f"{classification['false_positive']}"
    )

    print(
        f"False Negatives     : "
        f"{classification['false_negative']}"
    )

    print(
        f"Precision           : "
        f"{classification['precision'] * 100:.2f}%"
    )

    print(
        f"Recall              : "
        f"{classification['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score            : "
        f"{classification['f1_score'] * 100:.2f}%"
    )

    print(
        f"Accuracy            : "
        f"{classification['accuracy'] * 100:.2f}%"
    )


    # ==================================================
    # DOCUMENT-LEVEL RETRIEVAL
    # ==================================================

    print("\n")
    print("DOCUMENT-LEVEL RETRIEVAL")
    print("-" * 70)

    print(
        f"Document Hit@1      : "
        f"{document_hit_at_1 * 100:.2f}%"
    )

    print(
        f"Document Hit@{TOP_K:<10}: "
        f"{document_hit_at_k * 100:.2f}%"
    )

    print(
        f"Document MRR        : "
        f"{document_mrr:.4f}"
    )


    # ==================================================
    # DOCUMENT RANK DISTRIBUTION
    # ==================================================

    print("\n")
    print("DOCUMENT RANK DISTRIBUTION")
    print("-" * 70)

    print(
        f"Rank 1              : "
        f"{document_rank_distribution[1]}"
    )

    print(
        f"Rank 2              : "
        f"{document_rank_distribution[2]}"
    )

    print(
        f"Rank 3              : "
        f"{document_rank_distribution[3]}"
    )

    print(
        f"Not Retrieved       : "
        f"{document_rank_distribution['not_retrieved']}"
    )


    # ==================================================
    # CHUNK-LEVEL RETRIEVAL
    # ==================================================

    print("\n")
    print("CHUNK-LEVEL RETRIEVAL")
    print("-" * 70)

    print(
        f"Chunk Hit@1         : "
        f"{chunk_hit_at_1 * 100:.2f}%"
    )

    print(
        f"Chunk Hit@{TOP_K:<11}: "
        f"{chunk_hit_at_k * 100:.2f}%"
    )

    print(
        f"Chunk MRR           : "
        f"{chunk_mrr:.4f}"
    )


    # ==================================================
    # CHUNK RANK DISTRIBUTION
    # ==================================================

    print("\n")
    print("CHUNK RANK DISTRIBUTION")
    print("-" * 70)

    print(
        f"Rank 1              : "
        f"{chunk_rank_distribution[1]}"
    )

    print(
        f"Rank 2              : "
        f"{chunk_rank_distribution[2]}"
    )

    print(
        f"Rank 3              : "
        f"{chunk_rank_distribution[3]}"
    )

    print(
        f"Not Retrieved       : "
        f"{chunk_rank_distribution['not_retrieved']}"
    )


    # ==================================================
    # SOURCE DISTRIBUTION
    # ==================================================

    print("\n")
    print("SOURCE DISTRIBUTION")
    print("-" * 70)

    for document in sorted(
        source_distribution
    ):

        print(
            f"{document:<25}: "
            f"{source_distribution[document]}"
        )


    # ==================================================
    # QUERY RESULTS
    # ==================================================

    print("\n")
    print("=" * 70)
    print("QUERY RESULTS")
    print("=" * 70)


    for number, result in enumerate(
        evaluation_results,
        start=1
    ):

        status = (
            "PASS"
            if result["source_hit"]
            else "FAIL"
        )

        print(
            f"\n[{number:02d}] {status}"
        )

        print(
            f"Query: "
            f"{result['query']}"
        )

        print(
            f"Expected Relevant : "
            f"{result['expected_relevant']}"
        )

        print(
            f"Accepted          : "
            f"{result['accepted']}"
        )

        print(
            f"Best Distance     : "
            f"{result['best_distance']:.4f}"
        )

        print(
            f"Expected Documents: "
            f"{result['expected_documents']}"
        )

        print(
            f"Expected Sources  : "
            f"{result['expected_sources']}"
        )

        print(
            f"Retrieved Documents: "
            f"{result['retrieved_documents']}"
        )

        print(
            "Retrieved Sources:"
        )

        if not result["results"]:

            print(
                "  None"
            )

        else:

            for rank, chunk in enumerate(
                result["results"],
                start=1
            ):

                source = (
                    chunk.get(
                        "document",
                        "Unknown"
                    ),
                    chunk["chunk_id"]
                )

                expected_sources = set(
                    tuple(source)
                    for source in result.get(
                        "expected_sources",
                        []
                    )
                )

                chunk_match = (
                    source in expected_sources
                )

                match_label = (
                    " ← EXPECTED"
                    if chunk_match
                    else ""
                )

                print(
                    f"  Rank {rank}"
                    f" | "
                    f"{chunk.get('document', 'Unknown')}"
                    f" | Page "
                    f"{chunk['page']}"
                    f" | Chunk "
                    f"{chunk['chunk_id']}"
                    f" | Distance "
                    f"{chunk['distance']:.4f}"
                    f"{match_label}"
                )


    # ==================================================
    # COMPLETE
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "MULTI-DOCUMENT EVALUATION COMPLETE"
    )
    print("=" * 70)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()