# loads Text, Web, Directory, PDF, CSV, and JSON data source models, converts them into standard Document formats, and passes them into a pipeline powered by local HuggingFace embeddings (all-MiniLM-L6-v2) and a free Groq LLM model (llama-3.3-70b-versatile).

import os
import tempfile
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader,
    CSVLoader,
    JSONLoader,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


class MasterIngestionPipeline:
    """Manages ingestion across multiple file extensions and sources."""

    def __init__(self):
        self.all_documents: List[Document] = []

    def ingest_text(self, text_content: str, filename: str = "sample.txt"):
        """Ingests raw text via TextLoader."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
            tmp.write(text_content)
            tmp_path = tmp.name

        try:
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()

            # Retain clear original file names in metadata
            for d in docs:
                d.metadata["source"] = filename
            self.all_documents.extend(docs)
            print(f"[TextLoader] Loaded {len(docs)} document from {filename}")
        finally:
            os.remove(tmp_path)

    def ingest_url(self, url: str):
        """Ingests and clean web pages using WebBaseLoader."""
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            self.all_documents.extend(docs)
            print(f"[WebBaseLoader] Loaded {len(docs)} page from {url}")
        except Exception as e:
            print(f"[WebBaseLoader] Failed to load {url}: {e}")

    def ingest_pdf(self, pdf_path: str):
        """Ingests local PDF files using PyPDFLoader."""
        if not os.path.exists(pdf_path):
            print(f"[PyPDFLoader] File non-existent: {pdf_path}. Skipping.")
            return

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        self.all_documents.extend(docs)
        print(f"[PyPDFLoader] Loaded {len(docs)} page(s) from {pdf_path}")

    def ingest_csv(self, csv_content: str, filename: str = "data.csv"):
        """Ingests structured tabular rows into documents."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as tmp:
            tmp.write(csv_content)
            tmp_path = tmp.name

        try:
            loader = CSVLoader(file_path=tmp_path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = filename
            self.all_documents.extend(docs)
            print(
                f"[CSVLoader] Loaded {len(docs)} row record(s) from {filename}")
        finally:
            os.remove(tmp_path)

    def ingest_directory_lazy(self, dir_path: str):
        """Scans a target directory lazily to prevent RAM overload on huge batches."""
        if not os.path.exists(dir_path):
            return

        loader = DirectoryLoader(
            dir_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=False
        )

        count = 0
        for doc in loader.lazy_load():
            self.all_documents.append(doc)
            count += 1
        print(
            f"[DirectoryLoader] Lazily streamed {count} text documents from {dir_path}")


if __name__ == "__main__":
    pipeline = MasterIngestionPipeline()

    # 1. Load Text Document
    pipeline.ingest_text(
        text_content="System Rule: All active servers must have port 443 open for TLS connection.",
        filename="server_policy.txt"
    )

    # 2. Load Structured CSV Data
    csv_mock_data = "Error_Code,Description,Resolution\nERR_502,Bad Gateway,Restart nginx container\nERR_401,Unauthorized,Refresh OAuth2 token"
    pipeline.ingest_csv(csv_content=csv_mock_data, filename="error_codes.csv")

    # 3. Load Web Page
    pipeline.ingest_url("https://en.wikipedia.org/wiki/Web_scraping")

    # 4. Ingest PDF (If local demo PDF exists)
    pipeline.ingest_pdf("./docs/1302.4389v4.pdf")

    print("\n" + "=" * 60)
    print(
        f"TOTAL DOCUMENTS INGESTED IN PIPELINE: {len(pipeline.all_documents)}")
    print("=" * 60)

    # ----------VECTOR STORE AND RETRIEVAL -------------------------------
    print("\n1. Initializing Local HuggingFace Embeddings....")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("\n2. Storing Documents in Chroma VectorDB....")
    vectorstore = Chroma.from_documents(
        documents=pipeline.all_documents,
        embedding=embeddings,
        collection_name="master_loader_demo"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    print("\n3. Querying Vector Store with Free Groq LLM...")
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question accurately using only the provided context snippets."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    def query_loader_rag(question: str):
        docs = retriever.invoke(question)
        context = "\n---\n".join(
            [f"Source ({d.metadata.get('source', 'unknown')}): {d.page_content}" for d in docs])

        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})

        print(f"\nQUERY: '{question}'")
        print("RETRIEVED CONTEXT:")
        print(context[:300] + "...\n")
        print("ANSWER:")
        print(response.content)

    # Test query across CSV and Text Ingested Data
    query_loader_rag("What is the fix for ERR_502?")
    query_loader_rag("What port must be open for TLS?")
    query_loader_rag("What data does this csv contains")
    query_loader_rag("Tell me about webscraping")
    query_loader_rag("What is maxout network?")
    
    
#     load() vs lazy_load()
# When picking or customising document loaders, pay attention to the loading method:

# loader.load(): Reads all files/pages into memory at once and returns a single List[Document]. Ideal for smaller files or quick tasks.

# loader.lazy_load(): Returns a Python Generator. It yields one Document at a time. Always use lazy_load() for huge directories or multi-gigabyte files to keep memory usage low.
