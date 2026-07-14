import os
import tempfile
import numpy as np
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

load_dotenv()

# Model Dimensions: 384
print("Loading local HuggingFace embedding model...")
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def basic_embeddings():
    """Demonstrates single text generation and checking vector geometry properties."""
    text = "What is Machine Learning?"
    
    # Generate vector
    single_embedding = embeddings_model.embed_query(text)
    
    print("\n=== 1. Basic Embedding Analysis ===")
    print(f"Vector length (Dimensions): {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    # A norm of ~1.0 means it is a pre-normalized vector layout
    print(f"Vector geometric norm: {np.linalg.norm(single_embedding):.4f}")


def similarity_search():
    """Computes spatial similarity alignments manually via Cosine Similarity."""
    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]
    query = "What programming languages exist?"

    # Batch embed documents and single embed query
    doc_vectors = embeddings_model.embed_documents(docs)
    query_vector = embeddings_model.embed_query(query)

    # Cosine Similarity Formula: (A • B) / (||A|| * ||B||)
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vectors]
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print("\n=== 2. Semantic Similarity Evaluation ===")
    print(f"Query: '{query}'\n")
    print("Ranked Results:")
    for doc, score in ranked_docs:
        print(f"  [{score:.4f}] -> {doc}")


def embedding_caching():
    """Uses a local file store cache layer to bypass redundant embedding calculations."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tempdir:
        # Initialize an on-disk directory cache manager
        fs_store = LocalFileStore(root_path=tempdir)

        # Wrap our base model inside a CacheBackedEmbeddings manager
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings_model,
            document_embedding_cache=fs_store,
            namespace="rag_cache"
        )

        sample_text = ["What is Reinforcement Learning?"]

        print("\n=== 3. Embedding Cache In Action ===")
        
        # Call 1: Misses cache, computes using the underlying embedding model
        print("Call 1 (Computing baseline vectors)...")
        vectors_1 = cached_embeddings.embed_documents(sample_text)

        # Call 2: Hits cache, reads vector directly from disk
        print("Call 2 (Instant fetch from local disk cache)...")
        vectors_2 = cached_embeddings.embed_documents(sample_text)

        # Confirm identity matches
        match_verification = np.allclose(vectors_1[0], vectors_2[0])
        print(f"Vectors identical verification status: {match_verification}")


if __name__ == "__main__":
    # basic_embeddings()
    # similarity_search()
    embedding_caching()