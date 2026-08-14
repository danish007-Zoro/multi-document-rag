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

from multi_document_generation_evaluation import (
    TEST_QUERIES,
    evaluate_generation,
    calculate_generation_metrics
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
    print("MULTI-DOCUMENT RAG GENERATION EVALUATION")
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
    # RUN GENERATION EVALUATION
    # ==================================================

    print("\nRunning generation evaluation...")

    evaluation_results = evaluate_generation(
        TEST_QUERIES,
        pipeline,
        top_k=TOP_K,
        distance_threshold=DISTANCE_THRESHOLD
    )


    # ==================================================
    # CALCULATE METRICS
    # ==================================================

    metrics = calculate_generation_metrics(
        evaluation_results
    )


    # ==================================================
    # GENERATION QUALITY SUMMARY
    # ==================================================

    print("\n")
    print("=" * 70)
    print("GENERATION QUALITY SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal Queries          : "
        f"{metrics['total']}"
    )

    print(
        f"Passed                 : "
        f"{metrics['passed']}"
    )

    print(
        f"Failed                 : "
        f"{metrics['failed']}"
    )

    print(
        f"Overall Pass Rate      : "
        f"{metrics['pass_rate'] * 100:.2f}%"
    )

    print(
        f"Average Fact Coverage  : "
        f"{metrics['average_fact_coverage'] * 100:.2f}%"
    )

    print(
        f"Relevant Query Rate    : "
        f"{metrics['relevant_pass_rate'] * 100:.2f}%"
    )

    print(
        f"Refusal Accuracy       : "
        f"{metrics['refusal_accuracy'] * 100:.2f}%"
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
            if result["passed"]
            else "FAIL"
        )

        print(
            f"\n[{number:02d}] {status}"
        )

        print(
            f"Question: "
            f"{result['query']}"
        )

        print(
            f"Expected Relevant: "
            f"{result['expected_relevant']}"
        )

        print(
            f"Accepted: "
            f"{result['accepted']}"
        )

        print(
            f"Best Distance: "
            f"{result['best_distance']:.4f}"
        )

        print(
            f"Fact Coverage: "
            f"{result['fact_coverage'] * 100:.2f}%"
        )

        print(
            f"Matched Facts: "
            f"{result['matched_facts']}"
        )

        print(
            f"Missing Facts: "
            f"{result['missing_facts']}"
        )

        print(
            f"Source Hit: "
            f"{result['source_hit']}"
        )

        print(
            f"Retrieved Documents: "
            f"{result['retrieved_documents']}"
        )

        print("\nAnswer:")

        print(
            result["answer"]
        )

        print("\nSources:")

        if not result["results"]:

            print(
                "  None"
            )

        else:

            for chunk in result["results"]:

                print(
                    f"  "
                    f"{chunk.get('document', 'Unknown')}"
                    f" | Page "
                    f"{chunk['page']}"
                    f" | Chunk "
                    f"{chunk['chunk_id']}"
                    f" | Distance "
                    f"{chunk['distance']:.4f}"
                )


    # ==================================================
    # COMPLETE
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "MULTI-DOCUMENT GENERATION EVALUATION COMPLETE"
    )
    print("=" * 70)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()