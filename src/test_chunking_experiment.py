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

from extract_text import extract_text_from_pdf

from chunking import chunk_text

from embeddings import (
    load_embedding_model,
    generate_embeddings
)

from faiss_index import create_faiss_index

from multi_document_evaluation import (
    TEST_QUERIES,
    evaluate_document_attribution,
    calculate_document_hit_rate,
    calculate_document_hit_at_1,
    calculate_document_mrr,
    calculate_chunk_hit_rate,
    calculate_chunk_hit_at_1,
    calculate_chunk_mrr,
    calculate_classification_metrics
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
# CHUNK SIZE EXPERIMENT
#
# Overlap = 20% of chunk size
# ==================================================

CHUNK_SIZE_CONFIGS = [

    {
        "name": "Small",
        "chunk_size": 300,
        "overlap": 60
    },

    {
        "name": "Current",
        "chunk_size": 500,
        "overlap": 100
    },

    {
        "name": "Large",
        "chunk_size": 700,
        "overlap": 140
    },

    {
        "name": "Very Large",
        "chunk_size": 1000,
        "overlap": 200
    }
]


# ==================================================
# OVERLAP EXPERIMENT
#
# Chunk size fixed at current 500
# ==================================================

OVERLAP_CONFIGS = [

    {
        "name": "No Overlap",
        "chunk_size": 500,
        "overlap": 0
    },

    {
        "name": "Low Overlap",
        "chunk_size": 500,
        "overlap": 50
    },

    {
        "name": "Current",
        "chunk_size": 500,
        "overlap": 100
    },

    {
        "name": "High Overlap",
        "chunk_size": 500,
        "overlap": 150
    },

    {
        "name": "Very High Overlap",
        "chunk_size": 500,
        "overlap": 200
    }
]


# ==================================================
# BUILD INDEX
# ==================================================

def build_index_for_configuration(
    pdf_paths,
    model,
    chunk_size,
    overlap
):
    """
    Extract, chunk, embed and index all documents
    using one specific chunking configuration.
    """

    all_chunks = []

    total_pages = 0

    document_statistics = []


    # --------------------------------------------------
    # Process every document
    # --------------------------------------------------

    for pdf_path in pdf_paths:

        pages = extract_text_from_pdf(
            pdf_path
        )

        document_name = os.path.basename(
            pdf_path
        )

        total_pages += len(pages)


        # ----------------------------------------------
        # Chunk using experiment configuration
        # ----------------------------------------------

        document_chunks = chunk_text(
            pages,
            chunk_size=chunk_size,
            overlap=overlap
        )


        # ----------------------------------------------
        # Add document metadata
        # ----------------------------------------------

        for chunk in document_chunks:

            chunk["document"] = (
                document_name
            )


        all_chunks.extend(
            document_chunks
        )


        document_statistics.append(
            {
                "name": document_name,
                "pages": len(pages),
                "chunks": len(document_chunks)
            }
        )


    # --------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------

    all_chunks = generate_embeddings(
        all_chunks,
        model
    )


    # --------------------------------------------------
    # Create NumPy embedding matrix
    # --------------------------------------------------

    import numpy as np

    embeddings = np.array(
        [
            chunk["embedding"]
            for chunk in all_chunks
        ]
    ).astype(
        "float32"
    )


    # --------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------

    index = create_faiss_index(
        embeddings
    )


    return {
        "chunks": all_chunks,
        "index": index,
        "total_pages": total_pages,
        "total_chunks": len(all_chunks),
        "documents": document_statistics
    }


# ==================================================
# EVALUATE CONFIGURATION
# ==================================================

def evaluate_configuration(
    model,
    chunk_size,
    overlap
):
    """
    Build a complete retrieval index using the supplied
    chunking configuration and evaluate it against the
    fixed multi-document benchmark.
    """

    built = build_index_for_configuration(

        PDF_PATHS,

        model,

        chunk_size,

        overlap
    )


    # --------------------------------------------------
    # Run evaluation
    # --------------------------------------------------

    evaluation_results = (
        evaluate_document_attribution(

            TEST_QUERIES,

            model,

            built["index"],

            built["chunks"],

            top_k=TOP_K,

            distance_threshold=
                DISTANCE_THRESHOLD
        )
    )


    # --------------------------------------------------
    # Document metrics
    # --------------------------------------------------

    document_hit_1 = (
        calculate_document_hit_at_1(
            evaluation_results
        )
    )

    document_hit_3 = (
        calculate_document_hit_rate(
            evaluation_results,
            k=3
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

    chunk_hit_1 = (
        calculate_chunk_hit_at_1(
            evaluation_results
        )
    )

    chunk_hit_3 = (
        calculate_chunk_hit_rate(
            evaluation_results,
            k=3
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


    return {
        "chunks": built["total_chunks"],
        "pages": built["total_pages"],

        "document_hit_1":
            document_hit_1,

        "document_hit_3":
            document_hit_3,

        "document_mrr":
            document_mrr,

        "chunk_hit_1":
            chunk_hit_1,

        "chunk_hit_3":
            chunk_hit_3,

        "chunk_mrr":
            chunk_mrr,

        "classification":
            classification
    }


# ==================================================
# PRINT RESULT
# ==================================================

def print_result(
    config,
    result
):

    classification = (
        result["classification"]
    )


    print(
        f"\nConfiguration : "
        f"{config['name']}"
    )

    print(
        f"Chunk Size    : "
        f"{config['chunk_size']}"
    )

    print(
        f"Overlap       : "
        f"{config['overlap']}"
    )

    print(
        f"Total Chunks  : "
        f"{result['chunks']}"
    )

    print(
        f"Document Hit@1 : "
        f"{result['document_hit_1'] * 100:.2f}%"
    )

    print(
        f"Document Hit@3 : "
        f"{result['document_hit_3'] * 100:.2f}%"
    )

    print(
        f"Document MRR   : "
        f"{result['document_mrr']:.4f}"
    )

    print(
        f"Chunk Hit@1    : "
        f"{result['chunk_hit_1'] * 100:.2f}%"
    )

    print(
        f"Chunk Hit@3    : "
        f"{result['chunk_hit_3'] * 100:.2f}%"
    )

    print(
        f"Chunk MRR      : "
        f"{result['chunk_mrr']:.4f}"
    )

    print(
        f"Precision      : "
        f"{classification['precision'] * 100:.2f}%"
    )

    print(
        f"Recall         : "
        f"{classification['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score       : "
        f"{classification['f1_score'] * 100:.2f}%"
    )

    print(
        f"False Positives: "
        f"{classification['false_positive']}"
    )

    print(
        f"False Negatives: "
        f"{classification['false_negative']}"
    )


# ==================================================
# PRINT COMPARISON TABLE
# ==================================================

def print_comparison_table(
    results
):

    print("\n")

    print(
        f"{'Configuration':<18}"
        f"{'Size':>8}"
        f"{'Overlap':>10}"
        f"{'Chunks':>10}"
        f"{'Doc@1':>10}"
        f"{'Doc@3':>10}"
        f"{'Doc MRR':>10}"
        f"{'Chunk@1':>10}"
        f"{'Chunk@3':>10}"
        f"{'Chunk MRR':>10}"
    )

    print(
        "-" * 116
    )


    for item in results:

        print(
            f"{item['name']:<18}"
            f"{item['chunk_size']:>8}"
            f"{item['overlap']:>10}"
            f"{item['chunks']:>10}"
            f"{item['document_hit_1'] * 100:>9.2f}%"
            f"{item['document_hit_3'] * 100:>9.2f}%"
            f"{item['document_mrr']:>10.4f}"
            f"{item['chunk_hit_1'] * 100:>9.2f}%"
            f"{item['chunk_hit_3'] * 100:>9.2f}%"
            f"{item['chunk_mrr']:>10.4f}"
        )


# ==================================================
# FIND BEST CONFIGURATION
# ==================================================

def print_best_configuration(
    results,
    experiment_name
):

    print("\n")
    print("=" * 116)

    print(
        f"BEST {experiment_name.upper()} CONFIGURATION"
    )

    print("=" * 116)


    # --------------------------------------------------
    # Best document MRR
    # --------------------------------------------------

    best_document_mrr = max(
        results,
        key=lambda item:
            item["document_mrr"]
    )


    print(
        "\nBest Document MRR:"
    )

    print(
        f"  Configuration : "
        f"{best_document_mrr['name']}"
    )

    print(
        f"  Chunk Size    : "
        f"{best_document_mrr['chunk_size']}"
    )

    print(
        f"  Overlap       : "
        f"{best_document_mrr['overlap']}"
    )

    print(
        f"  Document MRR  : "
        f"{best_document_mrr['document_mrr']:.4f}"
    )


    # --------------------------------------------------
    # Best document Hit@3
    # --------------------------------------------------

    best_document_hit_3 = max(
        results,
        key=lambda item:
            item["document_hit_3"]
    )


    print(
        "\nBest Document Hit@3:"
    )

    print(
        f"  Configuration : "
        f"{best_document_hit_3['name']}"
    )

    print(
        f"  Score         : "
        f"{best_document_hit_3['document_hit_3'] * 100:.2f}%"
    )


    # --------------------------------------------------
    # Best chunk MRR
    # --------------------------------------------------

    best_chunk_mrr = max(
        results,
        key=lambda item:
            item["chunk_mrr"]
    )


    print(
        "\nBest Chunk MRR:"
    )

    print(
        f"  Configuration : "
        f"{best_chunk_mrr['name']}"
    )

    print(
        f"  Chunk MRR     : "
        f"{best_chunk_mrr['chunk_mrr']:.4f}"
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 116)

    print(
        "CHUNK SIZE AND OVERLAP EXPERIMENT"
    )

    print("=" * 116)


    print(
        "\nDocuments:"
    )

    for path in PDF_PATHS:

        print(
            f"  ✓ {os.path.basename(path)}"
        )


    print(
        f"\nQueries          : "
        f"{len(TEST_QUERIES)}"
    )

    print(
        f"Top-K            : "
        f"{TOP_K}"
    )

    print(
        f"Distance Threshold : "
        f"{DISTANCE_THRESHOLD}"
    )


    # ==================================================
    # VALIDATE DOCUMENTS
    # ==================================================

    print("\n")
    print("=" * 116)

    print(
        "VALIDATING DOCUMENTS"
    )

    print("=" * 116)


    for path in PDF_PATHS:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        print(
            f"✓ {os.path.basename(path)}"
        )


    # ==================================================
    # LOAD EMBEDDING MODEL ONCE
    # ==================================================

    print("\n")
    print("=" * 116)

    print(
        "LOADING EMBEDDING MODEL"
    )

    print("=" * 116)


    model = load_embedding_model()


    # ==================================================
    # EXPERIMENT A
    # ==================================================

    print("\n\n")

    print("=" * 116)

    print(
        "EXPERIMENT A: CHUNK SIZE"
    )

    print("=" * 116)

    print(
        "\nOverlap is fixed at approximately 20% "
        "of chunk size."
    )


    chunk_size_results = []


    for config in CHUNK_SIZE_CONFIGS:

        print("\n")
        print("-" * 116)

        print(
            f"Testing: {config['name']}"
        )

        print(
            f"Chunk Size = {config['chunk_size']} | "
            f"Overlap = {config['overlap']}"
        )

        print("-" * 116)


        result = evaluate_configuration(

            model,

            config["chunk_size"],

            config["overlap"]
        )


        result["name"] = (
            config["name"]
        )

        result["chunk_size"] = (
            config["chunk_size"]
        )

        result["overlap"] = (
            config["overlap"]
        )


        chunk_size_results.append(
            result
        )


        print_result(
            config,
            result
        )


    # ==================================================
    # CHUNK SIZE TABLE
    # ==================================================

    print("\n\n")

    print("=" * 116)

    print(
        "CHUNK SIZE COMPARISON"
    )

    print("=" * 116)

    print_comparison_table(
        chunk_size_results
    )


    print_best_configuration(
        chunk_size_results,
        "Chunk Size"
    )


    # ==================================================
    # EXPERIMENT B
    # ==================================================

    print("\n\n")

    print("=" * 116)

    print(
        "EXPERIMENT B: CHUNK OVERLAP"
    )

    print("=" * 116)

    print(
        "\nChunk size is fixed at 500 characters."
    )


    overlap_results = []


    for config in OVERLAP_CONFIGS:

        print("\n")
        print("-" * 116)

        print(
            f"Testing: {config['name']}"
        )

        print(
            f"Chunk Size = {config['chunk_size']} | "
            f"Overlap = {config['overlap']}"
        )

        print("-" * 116)


        result = evaluate_configuration(

            model,

            config["chunk_size"],

            config["overlap"]
        )


        result["name"] = (
            config["name"]
        )

        result["chunk_size"] = (
            config["chunk_size"]
        )

        result["overlap"] = (
            config["overlap"]
        )


        overlap_results.append(
            result
        )


        print_result(
            config,
            result
        )


    # ==================================================
    # OVERLAP TABLE
    # ==================================================

    print("\n\n")

    print("=" * 116)

    print(
        "OVERLAP COMPARISON"
    )

    print("=" * 116)

    print_comparison_table(
        overlap_results
    )


    print_best_configuration(
        overlap_results,
        "Overlap"
    )


    # ==================================================
    # CURRENT CONFIGURATION
    # ==================================================

    print("\n\n")

    print("=" * 116)

    print(
        "CURRENT CONFIGURATION"
    )

    print("=" * 116)


    current_configs = [

        result

        for result in chunk_size_results

        if (
            result["chunk_size"] == 500
            and
            result["overlap"] == 100
        )
    ]


    if current_configs:

        current = current_configs[0]

        print(
            f"\nChunk Size : "
            f"{current['chunk_size']}"
        )

        print(
            f"Overlap    : "
            f"{current['overlap']}"
        )

        print(
            f"Chunks     : "
            f"{current['chunks']}"
        )

        print(
            f"Document Hit@1 : "
            f"{current['document_hit_1'] * 100:.2f}%"
        )

        print(
            f"Document Hit@3 : "
            f"{current['document_hit_3'] * 100:.2f}%"
        )

        print(
            f"Document MRR   : "
            f"{current['document_mrr']:.4f}"
        )

        print(
            f"Chunk Hit@1    : "
            f"{current['chunk_hit_1'] * 100:.2f}%"
        )

        print(
            f"Chunk Hit@3    : "
            f"{current['chunk_hit_3'] * 100:.2f}%"
        )

        print(
            f"Chunk MRR      : "
            f"{current['chunk_mrr']:.4f}"
        )


    # ==================================================
    # FINAL INTERPRETATION
    # ==================================================

    print("\n\n")

    print("=" * 116)

    print(
        "EXPERIMENT INTERPRETATION"
    )

    print("=" * 116)


    print(
        "\nThe goal is to determine whether the current "
        "500-character / 100-character-overlap "
        "configuration is justified by retrieval results."
    )


    print(
        "\nExperiment A isolates the effect of chunk size "
        "while keeping overlap proportional."
    )


    print(
        "\nExperiment B isolates the effect of overlap "
        "while keeping chunk size fixed."
    )


    print(
        "\nDo not automatically choose the configuration "
        "with the most chunks."
    )


    print(
        "\nThe preferred configuration should provide "
        "strong retrieval metrics without unnecessarily "
        "fragmenting the documents."
    )


    # ==================================================
    # COMPLETE
    # ==================================================

    print("\n")

    print("=" * 116)

    print(
        "CHUNK SIZE AND OVERLAP EXPERIMENT COMPLETE"
    )

    print("=" * 116)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()