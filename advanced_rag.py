import os
import logging
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

print("Loading local HuggingFace embedding model (all-MiniLM-L6-v2)...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Connecting to ChatGroq LLaMA 3.3 platform...")
llm = init_chat_model("groq:llama-3.3-70b-versatile", temperature=0.0)


INFO_BURIED = [
    Document(
        page_content="""ACME AI SOLUTIONS - COMPANY HISTORY AND TECHNOLOGY STACK
Founded in 2018 by three Stanford graduates, ACME AI Solutions began as an enterprise ML consulting firm.
Our current technology stack has evolved significantly over the years. For backend services, we use Python and FastAPI.
Our data pipeline runs on Apache Spark and Airflow. For frontend, we've standardized on React and TypeScript.
LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory.
Our revenue has grown consistently, from $2M in 2019 to $45M in 2023. We project $70M for 2024.
The company went through Series B funding in 2022, raising $80M at a $500M valuation.""",
        metadata={"source": "acme_company_overview.pdf"},
    ),
    Document(
        page_content="""ACME AI PLATFORM - TECHNICAL DOCUMENTATION v2.4
The ACME AI Platform is built on a microservices architecture deployed on AWS EKS (Elastic Kubernetes Service).
Each microservice is containerized using Docker and orchestrated by Kubernetes. We use Istio as our service mesh.
Our database layer consists of PostgreSQL for transactional data, Redis for caching, and Pinecone for vector storage.
User authentication is handled through Auth0, supporting both SSO via SAML 2.0 and OAuth 2.0 flows.
LangGraph is a library for building stateful, multi-actor applications with LLMs using graphs.
We use DataDog for performance monitoring. Alert thresholds are configured for tail latency (p99 > 500ms).
Disaster recovery plan includes daily database backups stored in S3. RTO is 4 hours, and RPO is 1 hour.""",
        metadata={"source": "technical_docs_v2.4.pdf"},
    ),
]


def create_base_vectorstore(collection_name="base_advanced_demo"):
    """Helper to initialize an in-memory vector database instance."""
    return Chroma.from_documents(
        documents=INFO_BURIED,
        embedding=embeddings_model,
        collection_name=collection_name
    )
    
# Advanced Patterns Implementation

def multi_query_retriever():
    """Pattern 1: Generates alternative perspectives to maximize chunk recall."""
    print("\n" + "=" * 60)
    print("1. MULTI-QUERY RETRIEVER")
    print("=" * 60)
    
    vectorstore = create_base_vectorstore("multi_query_db")
    
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        llm=llm
    )
    
    query = "What tools are used to construct their core AI systems?"
    print(f"Original User Query: '{query}'\n")
    
    retrieved_docs = retriever.invoke(query)
    
    print(f"\nRetrieved {len(retrieved_docs)} Unique Chunks Across Expanded Queries:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"  [{i}] Source ({doc.metadata.get('source')}): {doc.page_content[:90]}...")
        
        
        
def contextual_compression():
    """Pattern 2: Minimizes context length by extracting only relevant sentences."""
    print("\n" + "=" * 60)
    print("2. CONTEXTUAL COMPRESSION RETRIEVER")
    print("=" * 60)
    
    vectorstore = create_base_vectorstore("compression_db")
    
    # initialize the LLM sentence-extractor filter
    compressor = LLMChainExtractor.from_llm(llm)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    )
    
    query = "What databases are configured in their cluster architecture?"
    print(f"Query: '{query}'\n")
        
    # Compare Full Chunk vs Compressed Output
    base_docs = vectorstore.as_retriever(search_kwargs={"k": 1}).invoke(query)
    print(f"--- Raw Full Chunk Content ({len(base_docs[0].page_content)} chars) ---")
    print(base_docs[0].page_content)

    compressed_docs = compression_retriever.invoke(query)
    print(f"\n--- Compressed Context Content ({len(compressed_docs[0].page_content)} chars) ---")
    print(compressed_docs[0].page_content)
        
        

def ensemble_hybrid_search():
    """Pattern 3: Combines Sparse Keyword (BM25) and Semantic Dense Search via RRF."""
    print("\n" + "=" * 60)
    print("3. ENSEMBLE HYBRID RETRIEVER")
    print("=" * 60)
    
    vectorstore = create_base_vectorstore("ensemble_db")
    
    # Sparse Keyword Engine
    bm25_retriever = BM25Retriever.from_documents(INFO_BURIED)
    bm25_retriever.k = 2
    
    # Dense Semantic Engine
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k":2})
    
    # Combine using alpha weights (40% keyword frequency match, 60% semantic map alignment)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6],
    )

    query = "What is the p99 alert threshold metric value?"
    print(f"Target Query: '{query}'\n")

    results = ensemble_retriever.invoke(query)
    for idx, doc in enumerate(results, start=1):
        print(f"Top Combined Rank {idx} Match: {doc.page_content[:110]}...")
        
        
        
def parent_document_retriever():
    """Pattern 4: Small chunks are used for indexing, larger parents for context."""
    print("\n" + "=" * 60)
    print("4. PARENT DOCUMENT RETRIEVER")
    print("=" * 60)
    
    # Configure granular child split boundaries alongside stable parent context blocks
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=15)
    
    vectorstore = Chroma(collection_name="parent_child_retriever", embedding_function=embeddings_model)
    doc_store = InMemoryStore()
    
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=doc_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    parent_retriever.add_documents(INFO_BURIED)
    query = "Explain LangGraph capabilities."
    print(f"Query: '{query}'\n")

    # Regular search returns a tiny fragment
    child_match = vectorstore.similarity_search(query, k=1)
    print(f"--- Child Chunk Found (Precise Search Match) --- \n{child_match[0].page_content}\n")

    # Parent retriever returns the complete paragraph contextual layout automatically
    parent_match = parent_retriever.invoke(query)
    print(f"--- Parent Chunk Returned (Rich Context for LLM) --- \n{parent_match[0].page_content}")
    
    
def integrated_advanced_rag_chain():
    """Pattern 5: Unified Pipeline fusing Multi-Query + Compression + QA Generation."""
    print("\n" + "=" * 60)
    print("5. COMPLETE UNIFIED ADVANCED RAG PIPELINE")
    print("=" * 60)

    vectorstore = create_base_vectorstore("unified_chain_db")
    
    # Stage A: Expand query coverage
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        llm=llm
    )
    
    # Stage B: Squeeze context noise
    compressor = LLMChainExtractor.from_llm(llm)
    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_retriever
    )
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the question concisely based strictly on the provided context summary metrics.
        Context:
        {context}

        Question: {question}

        Answer:
        """
        )
    
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Construct the pipeline via LCEL Architecture
    rag_chain = (
        {"context": advanced_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    query = "What was the Series B funding valuation and technology stack?"
    print(f"Executing Complete Pipeline for Question: '{query}'")
    
    answer = rag_chain.invoke(query)
    print(f"\nPipeline Final Response:\n{answer}")
        
    
    
        
if __name__ == "__main__":
    # multi_query_retriever()
    # contextual_compression()
    # ensemble_hybrid_search()
    # parent_document_retriever()
    integrated_advanced_rag_chain()