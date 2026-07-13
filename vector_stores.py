import os
import tempfile
import shutil
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Initialize FREE Local HuggingFace Embeddings (Runs locally on CPU/GPU)
print("Loading local HuggingFace embedding model...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# Core Evaluation Documents
SAMPLE_DOCS = [
    Document(page_content="LangChain is a framework for developing applications powered by language models.",
             metadata={"source": "langchain_docs", "topic": "overview"}),
    Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
             metadata={"source": "langgraph_docs", "topic": "overview"}),
    Document(page_content="Vector stores are databases optimized for storing and searching embeddings.",
             metadata={"source": "vector_guide", "topic": "database"}),
    Document(page_content="RAG combines retrieval with generation for more accurate LLM responses.",
             metadata={"source": "rag_guide", "topic": "architecture"}),
    Document(page_content="Embeddings convert text into numerical vectors for semantic similarity.",
             metadata={"source": "embeddings_guide", "topic": "fundamentals"}),
    Document(page_content="Chroma is an open-source embedding database for AI applications.",
             metadata={"source": "chroma_docs", "topic": "database"}),
    Document(page_content="FAISS is a library for efficient similarity search developed by Facebook.",
             metadata={"source": "faiss_docs", "topic": "database"}),
    Document(page_content="Pinecone is a managed vector database service for production workloads.",
             metadata={"source": "pinecone_docs", "topic": "database"}),
]


def chroma_basics():
    """Demonstrates standard vector ingestion and raw semantic similarity matching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )
        print(
            f"\n[Basics] Vector store initialized with {vectorstore._collection.count()} documents.")

        query = "What is LangChain?"
        results = vectorstore.similarity_search(query, k=2)

        print(f"Top 2 results for query '{query}':")
        for i, doc in enumerate(results):
            print(
                f"  Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

    # --- FIX FOR WINDOWS FILE LOCKING ---
        # 1. Close underlying client connection
        if hasattr(vectorstore, "_client") and hasattr(vectorstore._client, "close"):
            vectorstore._client.close()

        # 2. Delete instance reference to release handles
        del vectorstore


def similarity_search_with_scores():
    """Shows how to calculate similarity confidence from raw distance metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )

        query = "Explain vector stores."
        results_with_scores = vectorstore.similarity_search_with_score(
            query, k=3)

        print(f"\n[Scores] Top 3 results with scores for query '{query}':")
        for i, (doc, score) in enumerate(results_with_scores):
            # Convert raw L2 distance metric to similarity percentage
            final_score = 1 / (1 + score)
            print(
                f"  Result {i+1}: {doc.page_content} (Similarity: {final_score:.4f}, Distance: {score:.4f})")

    # --- FIX FOR WINDOWS FILE LOCKING ---
        # 1. Close underlying client connection
        if hasattr(vectorstore, "_client") and hasattr(vectorstore._client, "close"):
            vectorstore._client.close()

        # 2. Delete instance reference to release handles
        del vectorstore


def metadata_filtering():
    """Compares unrestricted vector searches against structured metadata constraints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )

        query = "What databases are available?"

        # 1. Broad Search
        results = vectorstore.similarity_search(query, k=3)
        print(f"\n[Filter] Unrestricted results for '{query}':")
        for i, doc in enumerate(results):
            print(
                f"  Result {i+1}: {doc.page_content} (Topic: {doc.metadata.get('topic')})")

        # 2. Hard Attribute Filter
        filter_criteria = {"topic": "database"}
        filtered_results = vectorstore.similarity_search(
            query, k=3, filter=filter_criteria)
        print(f"Filtered results (Topic == database) for '{query}':")
        for i, doc in enumerate(filtered_results):
            print(
                f"  Result {i+1}: {doc.page_content} (Topic: {doc.metadata.get('topic')})")

    # --- FIX FOR WINDOWS FILE LOCKING ---
        # 1. Close underlying client connection
        if hasattr(vectorstore, "_client") and hasattr(vectorstore._client, "close"):
            vectorstore._client.close()

        # 2. Delete instance reference to release handles
        del vectorstore


def as_retriever_modes():
    """Compares the output of standard Similarity algorithms vs Diversified MMR selection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )

        # Standard Retriever Mode
        sim_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3})
        sim_docs = sim_retriever.invoke("vector databases and embeddings")

        print("\n[Retriever] Standard Similarity Mode Results:")
        for i, doc in enumerate(sim_docs):
            print(f"  Result {i+1}: {doc.page_content}")

        # Diversified MMR Mode
        mmr_retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})
        mmr_docs = mmr_retriever.invoke("vector databases and embeddings")

        print("MMR Mode Results (Enforces Diversification):")
        for i, doc in enumerate(mmr_docs):
            print(f"  Result {i+1}: {doc.page_content}")

    # --- FIX FOR WINDOWS FILE LOCKING ---
        # 1. Close underlying client connection
        if hasattr(vectorstore, "_client") and hasattr(vectorstore._client, "close"):
            vectorstore._client.close()

        # 2. Delete instance reference to release handles
        del vectorstore


def persist_chroma_to_disk():
    """Demonstrates saving state to a local folder and reinstantiating from disk."""
    persist_dir = "./chroma_db_storage/"

    # Clean target folder path if it already exists from previous runs
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=persist_dir
    )
    print(
        f"\n[Persistence] Saved {vectorstore._collection.count()} items to disk at '{persist_dir}'")

    # Clear memory instance
    del vectorstore

    # Reload from disk
    reloaded_store = Chroma(
        embedding_function=embeddings_model, persist_directory=persist_dir)
    print(
        f"Reloaded instance successfully. Count: {reloaded_store._collection.count()} items.")

   


if __name__ == "__main__":
    chroma_basics()
    similarity_search_with_scores()
    metadata_filtering()
    as_retriever_modes()
    persist_chroma_to_disk()


# 1. Distance Metrics & Similarity Scores
# When we run similarity_search_with_score(), Chroma returns the raw mathematical distance between the query vector and the document vector.By default, Chroma uses Squared L2 Distance (Euclidean Distance).A distance of 0.0 means the vectors are perfectly identical.As the text becomes less semantically related, the distance score increases.
#
# Similarity Score = 1/{1 + Distance}.
# If distance is 0, similarity is 1.00 (100%).
# If distance is 0.5, similarity is 0.66 (66%).


# 2. Metadata Filtering
# Vector stores allow us to attach structural fields (like source, topic, author) to a text vector. Metadata Filtering happens before or during the vector search. Instead of searching all 10 million vectors in a production database, we instruct the store to discard anything that doesn't match {"topic": "database"} first, instantly speeding up execution and narrowing down accurate context.

# 3. Search Types: Similarity vs. MMR (Maximal Marginal Relevance)
# When converting a vector store into a retriever interface (vectorstore.as_retriever()), we can change how results are gathered:

#     search_type="similarity": Returns the absolute top-k closest matches. If we ask about "vector databases", it might return 3 chunks that say almost the exact same thing, creating redundant context.

#     search_type="mmr" (Maximal Marginal Relevance): First fetches a larger pool of documents (fetch_k=5), then optimizes for a balance between similarity to the query and diversity among the selected results. This prevents your LLM from reading repetitive sentences.
