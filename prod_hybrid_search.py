import os
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


def hybrid_retrieve(query: str, retrievers: list, weights: List[float], k: int = 3, rrf_k: int = 60) -> List[Document]:
    """Combines results from multiple retrievers using weighted Reciprocal Rank Fusion (RRF)."""
    doc_scores = {}  # Maps page_content string -> [cumulative_score, Document_object]

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)

        for rank, doc in enumerate(results):
            key = doc.page_content

            # RRF Formula: Weight * (1 / (rank + rrf_k))
            rrf_score = weight * (1.0 / (rank + rrf_k))

            if key in doc_scores:
                doc_scores[key][0] += rrf_score
            else:
                doc_scores[key] = [rrf_score, doc]

    # Sort documents based on the highest RRF score
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)

    return [doc for score, doc in sorted_docs[:k]]


class HybridRetriever:
    """Production hybrid retriever using HuggingFace Local Embeddings + BM25"""

    def __init__(self, documents: List[Document], bm25_weight: float = 0.5, k: int = 4):
        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1.0 - bm25_weight

        # 1. LOCAL EMBEDDINGS
        print("Loading HuggingFace Embedding Model....")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2. Vector store Initialization
        self.vectorstore = Chroma.from_documents(
            documents,
            self.embeddings,
            collection_name="free_hybrid_search"
        )
        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k})

        # 3. BM25 Sparse Search Initialization
        self.bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    def search(self, query: str) -> List[Document]:
        """Runs hybrid search using weighted RRF."""
        return hybrid_retrieve(
            query=query,
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[self.bm25_weight, self.vector_weight],
            k=self.k,
        )

    def add_documents(self, documents: List[Document]):
        """Add new documents increamentally."""
        self.vectorstore.add_documents(documents)

        # Sync BM25 by re-building from vectorstore contents
        all_data = self.vectorstore.get()
        all_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_data["documents"], all_data["metadatas"])
        ]
        self.bm25_retriever = BM25Retriever.from_documents(all_docs, k=self.k)


# --- FULL RAG PIPELINE EXECUTION ---
if __name__ == "__main__":
    # Test Corpus
    sample_documents = [
        Document(page_content="Product SKU-7742X is our flagship enterprise router. It supports gigabit speeds and advanced hardware acceleration."),
        Document(page_content="Router configuration guide: Access the admin panel dashboard at 192.168.1.1 to modify security settings."),
        Document(page_content="The authentication process requires valid credentials. Use OAuth2 tokens for secure API communication."),
        Document(page_content="WCAG 2.1 compliance requires all web images to have descriptive alt text and sufficient color contrast."),
        Document(page_content="Error code E_CONN_REFUSED indicates the target server rejected the TCP connection. Check firewall settings.")
    ]

    # Initialize Hybrid Retriever
    retriever = HybridRetriever(sample_documents, bm25_weight=0.5, k=3)

    # LLM Via Groq
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.0
    )

    # Prompt Template for RAG
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a precise technical support assistant. Answer the user's question using only the provided context."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    def run_rag_chain(query: str):
        print(f"\n{'='*20} Processing Query: '{query}' {'='*20}")

        # Step 1: Retrieve context using Hybrid Search
        retrieved_docs = retriever.search(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        print("\n--- Retrieved Context (RRF Merged) ---")
        for doc in retrieved_docs:
            print(f"- {doc.page_content}")

        # Step 2: Generate answer using Groq LLaMA 3
        chain = prompt_template | llm
        response = chain.invoke({"context": context, "question": query})

        print("\n--- LLM Response (Groq LLaMA 3.3) ---")
        print(response.content)

    # Test exact keyword, error code, and semantic queries
    run_rag_chain("What are the specs for SKU-7742X?")
    run_rag_chain("What does E_CONN_REFUSED mean?")


# The add_documents Catch

# Vector indices like Chroma allow you to incrementally stream new chunks seamlessly using .add_documents(). However, standard BM25 requires global corpus term statistics to accurately calculate inverse document frequencies ($IDF$).

# As implemented above, adding documents requires fetching existing strings from the vector index and re-initializing the BM25Retriever to keep both pipelines synchronized.
