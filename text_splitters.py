import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# --- SAMPLE DATASETS ---
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """Quicksort implementation in Python."""
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """Binary search implementation."""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''


def recursive_splitter():
    """Demonstrates standard recursive text chunking."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(SAMPLE_TEXT)

    print("\n=== 1. Recursive Character Text Splitter ===")
    print(f"Original Length: {len(SAMPLE_TEXT)} chars")
    print(f"Total Chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1} ({len(c)} chars): {c[:60].replace('\n', ' ')}...")


def overlap_importance():
    """Illustrates how overlap prevents contextual loss at chunk boundaries."""
    text = "Sentence 1: Machine learning is powerful. Sentence 2: Deep learning uses neural networks. Sentence 3: Transformers power modern LLMs."

    no_overlap = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=0)
    with_overlap = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=25)

    print("\n=== 2. Overlap Importance ===")
    print("NO OVERLAP:")
    for i, c in enumerate(no_overlap.split_text(text)):
        print(f"  Chunk {i+1}: {c}")

    print("\nWITH OVERLAP (25 chars):")
    for i, c in enumerate(with_overlap.split_text(text)):
        print(f"  Chunk {i+1}: {c}")


def markdown_splitter():
    """Demonstrates structure-aware header extraction into metadata."""
    headers_to_consider = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_consider)
    chunks = splitter.split_text(SAMPLE_TEXT)

    print("\n=== 3. Markdown Header Splitter ===")
    print(f"Produced {len(chunks)} header-bound chunks.\n")
    for i, chunk in enumerate(chunks[:2]):
        print(f"Chunk {i+1} Metadata: {chunk.metadata}")
        print(f"Chunk {i+1} Content snippet: {chunk.page_content[:100].replace('\n', ' ')}...\n")


def code_splitter():
    """Splits Python code at functional/class boundaries."""
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=250, chunk_overlap=30
    )
    chunks = python_splitter.split_text(SAMPLE_CODE)

    print("\n=== 4. Python Code Splitter ===")
    print(f"Produced {len(chunks)} code chunks.")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk.strip())


def pdf_document_splitter():
    """Splits full Document objects loaded from disk."""
    pdf_path = Path("./docs/Rohit_Gupta.pdf")
    
    print("\n=== 5. PDF Document Object Splitter ===")
    if not pdf_path.exists():
        print(f"File {pdf_path} not found. Skipping PDF execution.")
        return

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    print(f"Loaded Pages: {len(docs)} | Split Chunks: {len(split_docs)}")
    print(f"First Chunk Metadata: {split_docs[0].metadata}")
    print(f"First Chunk Snippet: {split_docs[0].page_content[:120].replace('\n', ' ')}...")
    print(f"Second Chunk Snippet: {split_docs[1].page_content[:180].replace('\n', ' ')}...")


if __name__ == "__main__":
    # recursive_splitter()
    # overlap_importance()
    # markdown_splitter()
    # code_splitter()
    pdf_document_splitter()