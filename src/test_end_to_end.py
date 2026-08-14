# ==================================================
# END-TO-END RAG PIPELINE TEST
# ==================================================

import os
import sys


# ==================================================
# PATH SETUP
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
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
# DOCUMENT PATHS
# ==================================================

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)

DOCUMENTS = [
    os.path.join(
        DATA_DIR,
        "Sample1.pdf"
    ),
    os.path.join(
        DATA_DIR,
        "Sample2.pdf"
    ),
    os.path.join(
        DATA_DIR,
        "Sample3.pdf"
    )
]


# ==================================================
# VALIDATE DOCUMENTS
# ==================================================

def validate_documents():

    print("=" * 70)
    print("END-TO-END RAG PIPELINE TEST")
    print("=" * 70)

    print()
    print("Validating documents...")
    print()

    for path in DOCUMENTS:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        print(
            f"✓ {os.path.basename(path)}"
        )


# ==================================================
# BUILD PIPELINE
# ==================================================

def build_pipeline():

    print()
    print("=" * 70)
    print("BUILDING RAG PIPELINE")
    print("=" * 70)
    print()

    pipeline = RAGPipeline(
        distance_threshold=1.4,
        top_k=3
    )

    print("Loading documents...")
    print()

    stats = pipeline.load_documents(
        DOCUMENTS
    )

    return pipeline, stats


# ==================================================
# PRINT PIPELINE STATISTICS
# ==================================================

def print_statistics(stats):

    print()
    print("=" * 70)
    print("PIPELINE STATISTICS")
    print("=" * 70)
    print()

    documents = stats.get(
        "documents",
        []
    )

    print(
        f"Documents : "
        f"{stats.get('total_documents', len(documents))}"
    )

    print(
        f"Pages     : "
        f"{stats.get('total_pages', 'N/A')}"
    )

    print(
        f"Chunks    : "
        f"{stats.get('total_chunks', 'N/A')}"
    )

    print(
        f"Embedding : "
        f"{stats.get('embedding_dimension', 'N/A')}"
    )

    print(
        f"FAISS     : "
        f"{stats.get('vectors', 'N/A')}"
    )


# ==================================================
# RUN GENERATION EVALUATION
# ==================================================

def run_generation_evaluation(
    pipeline
):

    print()
    print("=" * 70)
    print("GENERATION EVALUATION")
    print("=" * 70)
    print()

    results = evaluate_generation(
        TEST_QUERIES,
        pipeline,
        top_k=3,
        distance_threshold=1.4
    )

    metrics = calculate_generation_metrics(
        results
    )

    return results, metrics


# ==================================================
# PRINT RESULTS
# ==================================================

def print_results(
    results,
    metrics
):

    print()
    print("=" * 70)
    print("QUERY RESULTS")
    print("=" * 70)
    print()

    for index, result in enumerate(
        results,
        start=1
    ):

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"[{index:02d}] "
            f"{status} | "
            f"{result['query']}"
        )

        print(
            f"      Accepted       : "
            f"{result['accepted']}"
        )

        print(
            f"      Source Hit     : "
            f"{result['source_hit']}"
        )

        print(
            f"      Fact Coverage  : "
            f"{result['fact_coverage']:.2%}"
        )

        print(
            f"      Best Distance  : "
            f"{result['best_distance']:.4f}"
        )

        if result["missing_facts"]:

            print(
                f"      Missing Facts  : "
                f"{result['missing_facts']}"
            )

        print()

    print("=" * 70)
    print("END-TO-END EVALUATION SUMMARY")
    print("=" * 70)
    print()

    print(
        f"Total Queries          : "
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
        f"{metrics['pass_rate']:.2%}"
    )

    print(
        f"Average Fact Coverage  : "
        f"{metrics['average_fact_coverage']:.2%}"
    )

    print(
        f"Relevant Query Rate    : "
        f"{metrics['relevant_pass_rate']:.2%}"
    )

    print(
        f"Refusal Accuracy       : "
        f"{metrics['refusal_accuracy']:.2%}"
    )


# ==================================================
# VALIDATE RESULTS
# ==================================================

def validate_end_to_end_results(
    results,
    metrics
):

    failed_results = [
        result
        for result in results
        if not result["passed"]
    ]

    if failed_results:

        print()
        print("=" * 70)
        print("FAILED END-TO-END CASES")
        print("=" * 70)
        print()

        for result in failed_results:

            print(
                f"- {result['query']}"
            )

        raise AssertionError(
            f"{len(failed_results)} "
            f"end-to-end test case(s) failed."
        )

    if metrics["total"] == 0:

        raise AssertionError(
            "No end-to-end test cases were executed."
        )

    if metrics["pass_rate"] != 1.0:

        raise AssertionError(
            "End-to-end pass rate is below 100%."
        )

    print()
    print("=" * 70)
    print("✓ ALL END-TO-END TESTS PASSED")
    print("=" * 70)


# ==================================================
# MAIN
# ==================================================

def main():

    # ----------------------------------------------
    # 1. Validate documents
    # ----------------------------------------------

    validate_documents()


    # ----------------------------------------------
    # 2. Build RAG pipeline
    # ----------------------------------------------

    pipeline, stats = build_pipeline()


    # ----------------------------------------------
    # 3. Print statistics
    # ----------------------------------------------

    print_statistics(
        stats
    )


    # ----------------------------------------------
    # 4. Run generation evaluation
    # ----------------------------------------------

    results, metrics = (
        run_generation_evaluation(
            pipeline
        )
    )


    # ----------------------------------------------
    # 5. Print results
    # ----------------------------------------------

    print_results(
        results,
        metrics
    )


    # ----------------------------------------------
    # 6. Validate final result
    # ----------------------------------------------

    validate_end_to_end_results(
        results,
        metrics
    )


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    main()