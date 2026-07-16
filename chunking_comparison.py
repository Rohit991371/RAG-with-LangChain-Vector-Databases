# Recursive Character Text Splitter (Rule-Based): It is a top-down, character-count-first approach. It splits strictly on predefined structural separators: ["\n\n", "\n", " ", ""]. It aims to get as close to your target chunk_size as possible without cutting sentences in half.

# Semantic Chunker (AI-Based): It is a bottom-up, meaning-first approach. It splits the document into individual sentences, calculates the embedding vector for each sentence, and evaluates the cosine distance between consecutive sentences. If the semantic distance between Sentence A and Sentence B exceeds a calculated threshold (the breakpoint), it cuts the chunk there—regardless of the character count.

# -------------------------------------------------------

import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading local embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# ==========================================
# 1. TEST DATASETS
# ==========================================

STRUCTURED_DOC = """# System Architecture

We run our microservices inside a secure VPC. The gateway routes all external HTTPS traffic to our internal application load balancers.

## Database Configurations

Our database layer uses PostgreSQL with physical replication. We maintain a primary instance for read-write operations and two read replicas to scale query throughput.

## Security Policies

All environment variables must be encrypted at rest. We rotate IAM access keys every 90 days.
"""

UNSTRUCTURED_DOC = (
    "The deployment of our Kubernetes cluster completed successfully this morning. "
    "We verified that all pods are healthy and running. "
    "By the way, did you know that the first computer bug was a real moth found trapped in a relay by Grace Hopper in 1947? "
    "That is where the term debugging originates. "
    "Back to our production issues, we are currently experiencing a 2% packet loss on our database connection pool. "
    "We need to investigate the VPC subnet routing tables immediately to resolve this."
)


# ==========================================
# 2. RUN EVALUATOR
# ==========================================

def display_chunks(chunks: List[str], strategy_name: str):
    print(f"\n-------{strategy_name} (Total Chunks: {len(chunks)})----")
    for i, chunk in enumerate(chunks, start=1):
        cleaned_chunk = chunk.replace('\n', ' ').strip()
        print(
            f"  Chunk {i}  ({len(chunk)} chars): \"{cleaned_chunk[:120]}...\"")


def run_comparison(doc_text: str, doc_type: str):
    print("\n" + "=" * 80)
    print(f"EVALUATING: {doc_type}")
    print("=" * 80)

    # Configure Recursive Chunker
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""]
    )
    recursive_chunks = recursive_splitter.split_text(doc_text)
    display_chunks(recursive_chunks, "Recursive Character Splitter")

    # Configure Semantic Chunker (using 90th percentile of distance threshold)
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )
    semantic_chunks = semantic_splitter.split_text(doc_text)
    display_chunks(semantic_chunks, "Semantic Chunker (AI-Based)")


if __name__ == "__main__":
    run_comparison(STRUCTURED_DOC, "STRUCTUREED MARKDOWN DOCUMENT")
    run_comparison(UNSTRUCTURED_DOC, "UNSTRUCTURED CHAT / STREAM DOCUMENT")


# Case A: The Structured Markdown Document

# Recursive Chunking Winner: Recursive chunking excels here. Because the document has logical paragraph breaks (\n\n) and markdown headers, the recursive character splitter cleanly slices the file right at the structural boundaries (# System Architecture, ## Database Configurations, etc.).

# Semantic Downside: Because the sections are short and clear, semantic chunking is overkill. Running embeddings over every sentence introduces unnecessary local CPU/GPU latency just to find splits that regular expressions/new lines could have found instantly.

# -------------------------------------

# Case B: The Unstructured Stream Document

# Semantic Chunking Winner: The unstructured document shifts from Kubernetes Deployments --> The History of the first computer bug --> Database connection pool issues.

# Why Recursive Fails: A recursive character splitter with a target size of 150 characters will split the text blindly. It will package part of the Kubernetes sentence together with the historical computer bug sentence because they happen to sit next to each other within the character window limit. This dilutes the vector embedding space of that chunk.

# Why Semantic Shines: The Semantic Chunker plots these sentences in vector space and catches the radical mathematical transition:
#     Distance(Kubernetes, Grace Hopper Moth) >> Breakpoint Threshold

# It puts a hard chunk boundary right before the history lesson and another right after it. This ensures that the history chunk and the system deployment chunks remain entirely distinct, preventing irrelevant context retrieval later.
