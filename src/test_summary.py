import os
import sys

# Add src directory to Python path
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)

from rag_pipeline import RAGPipeline


pipeline = RAGPipeline()

stats = pipeline.load_documents(
    [
        "data/Sample1.pdf",
        "data/Sample2.pdf"
    ]
)

print("\n==============================")
print("DOCUMENTS LOADED")
print("==============================")

for document in stats["documents"]:

    print(
        f"{document['name']} | "
        f"{document['pages']} pages | "
        f"{document['chunks']} chunks"
    )


document_name = stats["documents"][0]["name"]

print("\n==============================")
print(f"SUMMARIZING: {document_name}")
print("==============================")

summary = pipeline.summarize(
    document_name
)

print("\n")
print(summary)