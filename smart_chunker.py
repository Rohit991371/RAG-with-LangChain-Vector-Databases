import os
import logging
from typing import List
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker


# Configure logs for production tracking
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - $(levelname)s - %(message)s")
logger = logging.getLogger("SmartChunker")

load_dotenv()

print("Loading local embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


def recursive_fallback(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """ Fallback utility using the standard Recursive Character Text Splitter. """
    logger.info(f"Applying Recursive Fallback with chunk_size={chunk_size}")
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return fallback_splitter.split_text(text)


def smart_chunker(
    text: str,
    max_chunk_size: int = 1000,
    fallback_chunk_size: int = 500,
    fallback_overlap: int = 50,
    breakpoint_percentile: int = 90
) -> List[str]:
    """
    Smarter hybrid chunking strategy:
    1. Attempts semantic chunking as primary.
    2. Validates that no individual chunk exceeds 'max_chunk_size'.
    3. If any chunk is oversized or the semantic splitter raises an exception, 
       gracefully falls back to Recursive Character splitting.
    """

    try:
        logger.info("Starting primary Semantic Chunking process....")

        # Initialize semantic chunker
        semantic_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_percentile
        )

        chunks = semantic_splitter.split_text(text)

        # Validate chunk boundaries
        oversized_chunks = [c for c in chunks if len(c) > max_chunk_size]

        if oversized_chunks:
            logger.warning(
                f"Semantic chunking returned {len(oversized_chunks)} oversized chunk(s) "
                f"excedding the {max_chunk_size} char threshold. Triggering fallback."
            )
            return recursive_fallback(text, fallback_chunk_size, fallback_overlap)

        logger.info(
            f"Semantic chunking successful! Returned {len(chunks)} balanced chunks.")

        return chunks

    except Exception as e:
        logger.error(
            f"Semantic chunking failed due to an error : {e}. Routing to fallback.")
        return recursive_fallback(text, fallback_chunk_size, fallback_overlap)


if __name__ == "__main__":
    # Test 1: Standard shifting text (Should pass semantic validation)
    standard_text = (
        "We successfully deployed the server update at midnight. "
        "All API gateways are fully functional. "
        "A completely different topic is that penguins live in the Southern Hemisphere and are flightless. "
        "Back to technology, our database connection pooling is optimized for high traffic."
    )

    # Test 2: Unbalanced, massive continuous text (Should fail chunk size validation and fallback)
    massive_unstructured_text = (
        "This is a very long paragraph that simulates a raw, unfiltered transcript. "
        "It contains highly repetitive and clustered semantic information. "
        "We are writing a massive continuous block of text here to force the semantic chunker to keep grouping sentences. "
        "Because the sentences look so similar semantically, the mathematical cosine difference between them is incredibly low. "
        "Therefore, no breakpoint is triggered, and it aggregates everything into a single massive block. "
        "If our max threshold is tight, this will be flagged as oversized and split using recursive rule structures."
    )

    print("\n" + "="*80)
    print("RUNNING TEST 1: SEMANTIC PASS")
    print("="*80)
    t1_chunks = smart_chunker(
        standard_text, max_chunk_size=1000, fallback_chunk_size=300)
    for idx, chunk in enumerate(t1_chunks, 1):
        print(f"  Chunk {idx} ({len(chunk)} chars): \"{chunk}\"")

    print("\n" + "="*80)
    print("RUNNING TEST 2: FALLBACK TRIGGERED")
    print("="*80)
    # Set a small max_chunk_size of 150 to guarantee a validation fallback
    t2_chunks = smart_chunker(
        massive_unstructured_text, max_chunk_size=150, fallback_chunk_size=100)
    for idx, chunk in enumerate(t2_chunks, 1):
        print(f"  Chunk {idx} ({len(chunk)} chars): \"{chunk}\"")



# Why this "Smart Chunker" Strategy is Excellent

# Handles Edge Cases (No Breakpoints): If a document is highly uniform (e.g., a long legal contract with extremely similar sentence structures), the semantic chunker might calculate a flat similarity curve. If it doesn't cross the mathematical threshold (or "breakpoint"), it might return one massive chunk of 10,000 characters. The validator catches this and cuts it down.

# Graceful Error Handling: Deep learning models, API limits, or local PyTorch/transformers memory limits can occasionally fail. Having a lightweight Regex-based fallback (RecursiveCharacterTextSplitter) ensures the ingestion pipeline never crashes halfway through a 1,000-document sync.

# Hybrid Performance: It gives us the high-recall semantic separation for unstructured sections while maintaining hard boundary safety for long blocks of text.