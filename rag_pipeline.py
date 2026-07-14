import os
import tempfile
from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

load_dotenv()

# ----------Local Stack Configuration---------------------
print("Initializating local HuggingFace embedding model........")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLM via Groq provider using init_chat_model
llm = init_chat_model("groq:llama-3.3-70b-versatile", temperature=0.2)

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.
2. **Prompts**: Templates for structuring inputs to language models.
3. **Chains**: Sequences of calls to models and other components.
4. **Agents**: Systems that use LLMs to determine which actions to take.
5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-groq
Create your first chain in under 10 lines of code.
"""


def create_kb(tempdir: str):
    """Creates a temporary Chroma vector store from the knowledge base."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(
        page_content=KNOWLEDGE_BASE,
        metadata={"source": "langchain_knowledge_base.md"}
    )
    chunks = splitter.split_documents([doc])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempdir
    )

    return vector_store


def basic_rag():
    """1. Basic RAG Pipeline"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tempdir:
        vector_store = create_kb(tempdir)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 2})

        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context: {context}
            Question: {question}
            Answer in concise manner. If you don't know, say "I don't know"
            
            """
        )

        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        # LCEL Chain Assembly
        rag_chain = (
            {
                "context": retriever | format_docs, "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        print("\n Outpur: ")
        for q in ["What is LangChain?", "Who created LangChain?", "What is LangGraph used for?", "Who is the founder of OpenAI?"]:
            answer = rag_chain.invoke(q)
            print(f"Q: {q}\nA: {answer}\n")


def rag_with_sources():
    """RAG With Source Citations"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        vectorstore = create_kb(tmpdir)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context. Include which sources you used: {context}
            Question: {question}
            Answer (include sources):"
            
            """
        )

        def format_docs_with_sources(docs):
            formatted = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "unknown")
                formatted.append(
                    f"[{i+1}] Source {{source}}:\n{doc.page_content}")
            return "\n\n".join(formatted)

        rag_chain = (
            {"context": retriever | format_docs_with_sources,
                "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print("\n--- 2. RAG with Sources ---")
        answer = rag_chain.invoke("What are the core components of LangChain?")
        print("Q: What are the core components?\n")
        print(f"A: {answer}")


def rag_with_fallback():
    """ RAG with Strict Fallback Guardrails"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        vectorstore = create_kb(tmpdir)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        prompt = ChatPromptTemplate.from_template(
            """Answer the question based ONLY on the following context. 
            If the answer is not in the context, respond with: "I don't have information about that in my knowledge base."
            Context: {context}
            Question: {question}
            Answer:
            """
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print("\n--- 3. RAG with Fallback ---")
        questions = [
            "What is the pricing for LangSmith?",     # In knowledge base
            "What is the stock price of OpenAI?",     # Out of knowledge base
        ]

        for q in questions:
            answer = rag_chain.invoke(q)
            print(f"Q: {q}\nA: {answer}\n")


def structured_rag():
    """RAG returning Pydantic Objects"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        vectorstore = create_kb(tmpdir)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        class RAGResponse(BaseModel):
            answer: str = Field(
                description="The precise answer to the question")
            confidence: str = Field(
                description="Confidence level: high, medium, or low")
            sources_used: List[str] = Field(
                description="List of source files referenced")
            follow_up: str = Field(description="A relevent follow-up question")

            # Bind Pydantic output schema to Groq LLM
        structured_llm = llm.with_structured_output(RAGResponse)

        prompt = ChatPromptTemplate.from_template(
            """Based on the context below, answer the question accurately."
            Context: {context}
            Question: {question}
            """
            )
        def format_docs(docs):
            return "\n\n".join(
                f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}"
                for doc in docs
                )
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | structured_llm
        )

        print("\n--- 4. Structured RAG ---")
        result: RAGResponse = rag_chain.invoke("What is LangGraph?")

        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence}")
        print(f"Sources: {result.sources_used}")
        print(f"Follow-up Suggestion: {result.follow_up}")


if __name__ == "__main__":
    # basic_rag()
    # rag_with_sources()
    # rag_with_fallback()
    structured_rag()
