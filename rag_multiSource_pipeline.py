# When working with diverse file types like PDF, JSON, Markdown, and CSV, the secret to source attribution lies in two things:

# Metadata Standardization: Ensuring every chunk from every loader carries a uniform, explicit source, file_type, and location identifier(page_number, row_number, etc.).

# Attribution Method: Deciding whether you want programmatic attribution(guaranteed 100 % accurate, directly from retrieved documents) or LLM inline citation(where the LLM cites sources within its generated answer).

#   ┌──────────────┐
#   │ PDF, CSV,    │ ── > [Loaders] ── > [Metadata Normalization]
#   │ JSON, MD     │(source, file_type, location)
#   └──────────────┘                                    │
#                                                       ▼
#   ┌──────────────┐                             ┌──────────────┐
#   │ User Query   │ ── > [Vector Search] ─── > │  Retrieved   │
#   └──────────────┘                             │   Chunks     │
#                                                └──────┬───────┘
#                                                       │
#                          ┌────────────────────────────┴────────────────────────────┐
#                          ▼                                                         ▼
#          [ Method 1: Programmatic Attribution ]                     [ Method 2: LLM Structured Citation ]
#          (Direct metadata return alongside answer)                  (LLM outputs answer + sources array)

#-----------------------------------------------------------------------------------------------------------------------------------------

import os
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    JSONLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

load_dotenv()

# Set local docs folder path
BASE_DIR = Path(__file__).resolve().parent

# 2. Append your subfolder structure dynamically
DOCS_DIR = BASE_DIR / "docs"

# ==========================================
# 1. MODELS & SCHEMAS FOR METHOD 2
# ==========================================

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = init_chat_model("groq:llama-3.3-70b-versatile", temperature=0.0)


class SourceCitation(BaseModel):
    """Schema for individual document citations."""
    file_name: str = Field(description="The source file name (e.g. manual.pdf, products.csv)")
    file_type: str = Field(description="Format type: PDF, CSV, JSON, or Markdown")
    location: str = Field(description="Location marker: Page X, Row Y, Record #Z, or Section")


class RAGResponseWithSources(BaseModel):
    """Schema for Method 2 (Structured LLM Output)."""
    answer: str = Field(description="Comprehensive answer based on context")
    sources: List[SourceCitation] = Field(description="List of specific sources referenced")


# ==========================================
# 2. DOCUMENT LOADERS & METADATA NORMALIZATION
# ==========================================

def load_and_normalize_docs(directory_path: Path) -> List[Document]:
    """
    Scans the docs directory and standardizes metadata across
    PDF, CSV, JSON, and MD/TXT files.
    """
    normalized_docs: List[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    if not directory_path.exists():
        print(f"Directory {directory_path} does not exist!")
        return normalized_docs

    for file_path in directory_path.rglob("*"):
        if file_path.is_dir():
            continue

        ext = file_path.suffix.lower()
        file_name = file_path.name
        raw_docs = []

        try:
            # 1. PDF Files
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
                for doc in loader.load():
                    doc.metadata["file_name"] = file_name
                    doc.metadata["file_type"] = "PDF"
                    doc.metadata["location"] = f"Page {doc.metadata.get('page', 0) + 1}"
                    raw_docs.append(doc)

            # 2. CSV Files
            elif ext == ".csv":
                loader = CSVLoader(str(file_path))
                for doc in loader.load():
                    doc.metadata["file_name"] = file_name
                    doc.metadata["file_type"] = "CSV"
                    doc.metadata["location"] = f"Row {doc.metadata.get('row', 'N/A')}"
                    raw_docs.append(doc)

            # 3. JSON Files
            elif ext == ".json":
                loader = JSONLoader(file_path=str(file_path), jq_schema=".[]", text_content=False)
                for i, doc in enumerate(loader.load()):
                    doc.metadata["file_name"] = file_name
                    doc.metadata["file_type"] = "JSON"
                    doc.metadata["location"] = f"Record #{i+1}"
                    raw_docs.append(doc)

            # 4. Text / Markdown Files
            elif ext in [".txt", ".md"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
                for doc in loader.load():
                    doc.metadata["file_name"] = file_name
                    doc.metadata["file_type"] = "Markdown" if ext == ".md" else "Text"
                    doc.metadata["location"] = "Full Document"
                    raw_docs.append(doc)

            if raw_docs:
                chunks = text_splitter.split_documents(raw_docs)
                normalized_docs.extend(chunks)
                print(f"Ingested {len(chunks)} chunks from {file_name} ({ext.upper()})")

        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    return normalized_docs


# ==========================================
# 3. RAG IMPLEMENTATIONS
# ==========================================

class DualAttributionRAG:
    def __init__(self, docs_directory: Path):
        print(f"Loading and normalizing files from: {docs_directory}\n" + "-" * 60)
        docs = load_and_normalize_docs(docs_directory)

        if not docs:
            raise ValueError(f"No valid documents found in {docs_directory}")

        print("-" * 60)
        print(f"Initializing Chroma Vector Store with {len(docs)} chunks...")
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name="docs_attribution_demo"
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Base Prompt for Method 1
        self.base_prompt = ChatPromptTemplate.from_template(
            """Answer the question concisely using ONLY the following context:

Context:
{context}

Question: {question}"""
        )

        # Structured Prompt for Method 2
        self.structured_prompt = ChatPromptTemplate.from_template(
            """Answer the question using ONLY the provided context snippets.
For every detail referenced in your answer, extract and cite its exact file name, file type, and location.

Context:
{context}

Question: {question}"""
        )

    # ----------------------------------------------------
    # METHOD 1: Programmatic Source Attribution
    # ----------------------------------------------------
    def run_method_1_programmatic(self, query: str) -> Dict[str, Any]:
        """
        Retrieves documents first, extracts metadata programmatically (100% accurate),
        and passes context to a standard string-output LLM chain.
        """
        # Step A: Retrieve relevant documents directly from vector store
        retrieved_docs = self.retriever.invoke(query)

        # Step B: Format context for standard LLM
        context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Step C: Generate text answer
        chain = self.base_prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "question": query})

        # Step D: Extract unique sources directly from metadata (Zero LLM Hallucination)
        sources_used = []
        seen = set()
        for doc in retrieved_docs:
            m = doc.metadata
            key = (m.get("file_name"), m.get("file_type"), m.get("location"))
            if key not in seen:
                seen.add(key)
                sources_used.append({
                    "file_name": m.get("file_name", "unknown"),
                    "file_type": m.get("file_type", "unknown"),
                    "location": m.get("location", "N/A")
                })

        return {
            "query": query,
            "answer": answer,
            "sources": sources_used
        }

    # ----------------------------------------------------
    # METHOD 2: LLM Structured Citation (Pydantic Output)
    # ----------------------------------------------------
    def run_method_2_structured(self, query: str) -> RAGResponseWithSources:
        """
        Injects normalized metadata headers directly into context and
        forces the LLM to output a structured JSON array of sources using Pydantic.
        """
        def format_context_with_metadata(docs: List[Document]) -> str:
            formatted = []
            for doc in docs:
                m = doc.metadata
                header = f"[FILE: {m.get('file_name')} | TYPE: {m.get('file_type')} | LOCATION: {m.get('location')}]"
                formatted.append(f"{header}\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted)

        structured_llm = llm.with_structured_output(RAGResponseWithSources)

        chain = (
            {
                "context": self.retriever | format_context_with_metadata,
                "question": RunnablePassthrough()
            }
            | self.structured_prompt
            | structured_llm
        )

        return chain.invoke(query)


# ==========================================
# 4. EXECUTION DEMO
# ==========================================

if __name__ == "__main__":
    rag_system = DualAttributionRAG(DOCS_DIR)

    # query = "What information is available in the documents?"
    # query = "Who is the person in the sources? And what information does the test house contains" # only answers about the house data but not person
    query = "Who is the person rohit in the sources? And what information does the test house contains" # Answers both

    print("\n" + "=" * 80)
    print("METHOD 1: PROGRAMMATIC ATTRIBUTION (Direct Metadata Extraction)")
    print("=" * 80)
    m1_result = rag_system.run_method_1_programmatic(query)
    print(f"\nAnswer:\n{m1_result['answer']}\n")
    print("Retrieved Sources (Direct from Vector Store):")
    for src in m1_result["sources"]:
        print(f"  • {src['file_name']} [{src['file_type']}] -> {src['location']}")

    print("\n" + "=" * 80)
    print("METHOD 2: LLM STRUCTURED CITATION (Pydantic Output)")
    print("=" * 80)
    m2_result: RAGResponseWithSources = rag_system.run_method_2_structured(query)
    print(f"\nAnswer:\n{m2_result.answer}\n")
    print("Cited Sources (Generated by Structured LLM):")
    for src in m2_result.sources:
        print(f"  • {src.file_name} [{src.file_type}] -> {src.location}")
        
        
        
        
"""
===================================================================================
CRITICAL NOTE ON VECTOR SEARCH & MULTI-TOPIC QUERIES (Embedding Dilution)
===================================================================================

1. How Vector Search Operates:
   - Vector stores convert the ENTIRE query into a single point (embedding) in 
     vector space.
   - It DOES NOT check file names or metadata first; it only measures mathematical 
     distance between the query embedding and chunk embeddings.

2. Query/Embedding Dilution:
   - Asking multi-part or unrelated questions in a single string 
     (e.g., "Who is the person? AND What does the test house contain?") dilutes the 
     query vector.
   - If one topic dominates semantically (e.g., "test house" matching many CSV rows), 
     all top-k slots (e.g., k=3) will be occupied by that single topic, completely 
     filtering out other relevant documents (like the person's PDF).

3. Solutions for Production RAG:
   - Keyword Anchoring: Adding specific keywords (e.g., "Who is Rohit...") strengthens 
     the vector signal for secondary documents.
   - Increase Top-K: Increase k (e.g., k=6 or k=10) to give multi-topic queries 
     enough retrieval slots across diverse sources.
   - Query Decomposition: Use an LLM step to split compound user prompts into 
     individual sub-queries before calling retriever.invoke().
===================================================================================
"""