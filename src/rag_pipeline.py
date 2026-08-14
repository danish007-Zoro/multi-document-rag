import os
import numpy as np

from extract_text import extract_text_from_pdf
from chunking import chunk_text
from embeddings import load_embedding_model, generate_embeddings
from faiss_index import create_faiss_index
from retrieval import search

from prompting import build_context, build_prompt
from generation import generate_answer
from summarization import summarize_document


class RAGPipeline:

    def __init__(
        self,
        distance_threshold=1.4,
        top_k=3
    ):
        """
        Initialize the RAG pipeline.

        Args:
            distance_threshold (float):
                Maximum FAISS distance considered relevant.

            top_k (int):
                Number of chunks retrieved from FAISS.
        """

        self.distance_threshold = distance_threshold
        self.top_k = top_k

        self.model = None
        self.index = None
        self.chunks = []

        self.documents = []
        self.document_pages = {}

        self.total_pages = 0

    def load_documents(self, pdf_paths):
        """
        Process multiple PDF documents and build one FAISS index.

        Pipeline:

        Multiple PDFs
              ↓
        Text extraction
              ↓
        Page storage
              ↓
        Chunking
              ↓
        Metadata
              ↓
        Embeddings
              ↓
        One FAISS index
        """

        if not pdf_paths:
            raise ValueError(
                "No PDF documents were provided."
            )

        # Load embedding model only once
        self.model = load_embedding_model()

        all_chunks = []
        total_pages = 0
        documents = []

        # Reset document-level page storage
        self.document_pages = {}

        # Process every PDF
        for pdf_path in pdf_paths:

            # ------------------------------------------
            # Extract text from PDF
            # ------------------------------------------

            pages = extract_text_from_pdf(
                pdf_path
            )

            # ------------------------------------------
            # Document name
            # ------------------------------------------

            document_name = os.path.basename(
                pdf_path
            )

            # ------------------------------------------
            # Store original pages
            #
            # Used later for document summarization.
            # ------------------------------------------

            self.document_pages[
                document_name
            ] = pages

            total_pages += len(pages)

            # ------------------------------------------
            # Create chunks for this document
            # ------------------------------------------

            document_chunks = chunk_text(
                pages
            )

            # ------------------------------------------
            # Add document metadata
            # ------------------------------------------

            for chunk in document_chunks:

                chunk["document"] = document_name

            all_chunks.extend(
                document_chunks
            )

            # ------------------------------------------
            # Store document statistics
            # ------------------------------------------

            documents.append(
                {
                    "name": document_name,
                    "pages": len(pages),
                    "chunks": len(document_chunks)
                }
            )

        # ----------------------------------------------
        # Generate embeddings for all chunks
        # ----------------------------------------------

        all_chunks = generate_embeddings(
            all_chunks,
            self.model
        )

        # ----------------------------------------------
        # Convert embeddings to NumPy matrix
        # ----------------------------------------------

        embeddings = np.array(
            [
                chunk["embedding"]
                for chunk in all_chunks
            ]
        ).astype("float32")

        # ----------------------------------------------
        # Create ONE FAISS index
        # ----------------------------------------------

        self.index = create_faiss_index(
            embeddings
        )

        # ----------------------------------------------
        # Store chunks
        # ----------------------------------------------

        self.chunks = all_chunks

        # ----------------------------------------------
        # Store document information
        # ----------------------------------------------

        self.documents = documents
        self.total_pages = total_pages

        return {
            "documents": documents,
            "total_documents": len(documents),
            "total_pages": total_pages,
            "total_chunks": len(all_chunks),
            "embedding_dimension": embeddings.shape[1],
            "vectors": self.index.ntotal
        }

    def load_document(self, pdf_path):
        """
        Backward-compatible single-document loader.

        This allows existing testing code to continue working.
        """

        return self.load_documents(
            [pdf_path]
        )

    def summarize(self, document_name):
        """
        Generate a summary for a loaded document.

        Args:
            document_name (str):
                Name of the document to summarize.

        Returns:
            str:
                Generated document summary.
        """

        if not self.document_pages:
            raise RuntimeError(
                "No documents have been loaded."
            )

        if document_name not in self.document_pages:

            raise ValueError(
                f"Document '{document_name}' was not found."
            )

        pages = self.document_pages[
            document_name
        ]

        return summarize_document(
            document_name,
            pages
        )

    def ask(self, query):
        """
        Ask a question about the loaded documents.

        Returns:
            dict containing answer, sources
            and retrieval information.
        """

        if self.model is None or self.index is None:

            raise RuntimeError(
                "No document has been loaded."
            )

        # ----------------------------------------------
        # Retrieve relevant chunks
        # ----------------------------------------------

        retrieval_result = search(
            query,
            self.model,
            self.index,
            self.chunks,
            top_k=self.top_k,
            distance_threshold=self.distance_threshold
        )

        # ----------------------------------------------
        # Reject if retrieval is not sufficiently relevant
        # ----------------------------------------------

        if not retrieval_result["accepted"]:

            return {
                "answer": (
                    "I could not find the answer "
                    "in the provided documents."
                ),

                "sources": [],

                "accepted": False,

                "best_distance": (
                    retrieval_result["best_distance"]
                ),

                "results": (
                    retrieval_result["results"]
                )
            }

        # ----------------------------------------------
        # Build context
        # ----------------------------------------------

        context = build_context(
            retrieval_result["results"]
        )

        # ----------------------------------------------
        # Build grounded prompt
        # ----------------------------------------------

        prompt = build_prompt(
            query,
            context
        )

        # ----------------------------------------------
        # Generate answer
        # ----------------------------------------------

        answer = generate_answer(
            prompt
        )

        # ----------------------------------------------
        # Extract source information
        # ----------------------------------------------

        sources = []

        for chunk in retrieval_result["results"]:

            sources.append(
                {
                    "document": chunk.get(
                        "document",
                        "Unknown"
                    ),

                    "page": chunk["page"],

                    "chunk_id": chunk["chunk_id"],

                    "distance": chunk["distance"]
                }
            )

        return {
            "answer": answer,

            "sources": sources,

            "accepted": True,

            "best_distance": (
                retrieval_result["best_distance"]
            ),

            "results": (
                retrieval_result["results"]
            )
        }