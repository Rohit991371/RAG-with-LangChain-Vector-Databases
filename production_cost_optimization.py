# User Query
#            │
#            ▼
# ┌──────────────────────┐      Cache Hit (Similarity >= 0.92)
# │ Vector Semantic Cache│ ────────────────────────────────────┐
# └──────────┬───────────┘                                     │
#            │ Cache Miss                                      │
#            ▼                                                 │
# ┌──────────────────────┐                                     │
# │ LLM Complexity Route │                                     │
# └──────────┬───────────┘                                     │
#            ├─────────────────────────┐                       │
#            ▼ (Simple)                ▼ (Complex)             │
# ┌─────────────────────┐    ┌─────────────────────────┐       │
# │ LLaMA-3.1-8B-Instant│    │ LLaMA-3.3-70B-Versatile │       │
# └──────────┬──────────┘    └─────────┬───────────────┘       │
#            │                         │                       │
#            └───────────┬─────────────┘                       │
#                        ▼                                     ▼
#            ┌──────────────────────┐              ┌──────────────────────┐
#            │ Update Vector Cache  │              │ Return Pre-Saved     │
#            │ & Track Real Tokens  │              │ Text Instant Response│
#            └──────────────────────┘              └──────────────────────┘

import os
import time
import logging
from typing import Tuple, Dict, Any, Optional
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage
from langsmith import traceable

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CostOptimizer")

load_dotenv()

# Set up LangSmith Observability parameters dynamically
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_PROJECT"] = os.getenv(
    "LANGSMITH_PROJECT", "cost_optimization_patterns")


# METRICS AND TELEMETRY TRACKING LAYER
class TokenTracker:
    """Tracks exact token metrics and calculates live financial expenditure infrastructure overhead."""

    def __init__(self):
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "routed_simple": 0,
            "routed_complex": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0
        }

        # Approximate production pricing metrics per 1M tokens
        self.pricing = {
            "llama-3.1-8b": {"input": 0.05, "output": 0.08},
            "llama-3.3-70b": {"input": 0.59, "output": 0.79}
        }
        self.accumulated_cost = 0.0

    def record_llm_usage(self, response_message: AIMessage, model_tier: str):
        """Extracts and logs precise token counters out of Groq API wire payloads."""
        token_usage = response_message.response_metadata.get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)

        self.stats["total_prompt_tokens"] += prompt_tokens
        self.stats["total_completion_tokens"] += completion_tokens

        # Calculate real financial expenditure base rates
        tier = "llama-3.3-70b" if model_tier == "complex" else "llama-3.1-8b"
        input_cost = (prompt_tokens / 1_000_000) * self.pricing[tier]["input"]
        output_cost = (completion_tokens / 1_000_000) * \
            self.pricing[tier]["output"]

        self.accumulated_cost += (input_cost + output_cost)


# VECTOR SEMANTIC CACHING ENGINE
class ChromaSemanticCache:
    """Maintains a semantic vector store to instantly resolve contextually similar queries."""

    def __init__(self, embedding_model, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        # Core collection mapping strings to previously resolved text fields
        self.vectorstore = Chroma(
            collection_name="semantic_llm_cache",
            embedding_function=embedding_model
        )

    def lookup(self, query: str) -> Optional[str]:
        """Performs vector distance searches to catch context matches above threshold limitations."""
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=1)
        if results:
            doc, score = results[0]
            if score >= self.threshold:
                logger.info(
                    f"Semantic Cache Hit! Confidence match score: {score:.4f}")
                return doc.metadata.get("cached_response")
        return None

    def update(self, query: str, response: str):
        """Indexes processed query vectors to protect down-stream execution lines."""
        self.vectorstore.add_texts(
            texts=[query],
            metadatas=[{"cached_response": response}]
        )


# 3. HIGH PERFORMANCE ROUTED RAG CORE
class ProductionCostOptimizedLLM:
    """Unified wrapper combining Semantic Caching and Dynamic Complexity Routing."""

    def __init__(self, cache_similarity: float = 0.92):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.cache = ChromaSemanticCache(
            self.embeddings, similarity_threshold=cache_similarity)
        self.tracker = TokenTracker()

        # Deploy models based on budget configurations
        self.classifier_llm = init_chat_model(
            "groq:llama-3.1-8b-instant", temperature=0.0)
        self.simple_llm = init_chat_model(
            "groq:llama-3.1-8b-instant", temperature=0.1)
        self.complex_llm = init_chat_model(
            "groq:llama-3.3-70b-versatile", temperature=0.0)

    @traceable(name="classify_route_complexity")
    def _route_query(self, query: str) -> str:
        """Classifies raw inbound prompts using rapid classification templates."""
        prompt = ChatPromptTemplate.from_template(
            "Classify this query complexity. Respond with exactly one word: 'simple' or 'complex'.\nQuery: {query}"
        )
        chain = prompt | self.classifier_llm | StrOutputParser()
        decision = chain.invoke({"query": query}).strip().lower()
        return "complex" if "complex" in decision else "simple"

    @traceable(name="optimized_llm_invocation")
    def invoke(self, query: str) -> str:
        # Step 1: Query the Vector Semantic Cache
        cached_result = self.cache.lookup(query)
        if cached_result:
            self.tracker.stats["cache_hits"] += 1
            return f"[RESOLVED VIA SEMANTIC CACHE] {cached_result}"

        self.tracker.stats["cache_misses"] += 1

        # Step 2: Dynamically classify query complexity mapping routes
        route_tier = self._route_query(query)

        if route_tier == "simple":
            self.tracker.stats["routed_simple"] += 1
            logger.info("Routing query to Low-Cost Engine (LLaMA-3.1-8B)...")
            response = self.simple_llm.invoke(query)
        else:
            self.tracker.stats["routed_complex"] += 1
            logger.info(
                "Escalating query to High-Reasoning Engine (LLaMA-3.3-70B)...")
            response = self.complex_llm.invoke(query)

        # Step 3: Extract exact token usage structures and update metrics logs
        self.tracker.record_llm_usage(response, route_tier)

        # Step 4: Write the newly processed answer into cache vectors
        self.cache.update(query, response.content)

        return response.content


if __name__ == "__main__":
    # Initialize optimization core pipeline configurations
    optimized_pipeline = ProductionCostOptimizedLLM(cache_similarity=0.90)

    test_runs = [
        # 1. Simple -> Hits 8B
        "What is 5 + 5?",
        # 2. Semantic Cache Hit (Bypasses LLM entirely)
        "What is 5 plus 5?",
        # 3. Complex -> Escalates to 70B
        "Analyze the cryptographic security differences between RSA and ECC keys.",
        # 4. Semantic Cache Hit (Bypasses LLM entirely)
        "Compare the security of cryptographic keys like RSA and ECC."
    ]

    print("\n--- Initializing Live Core Inbound Requests Pipeline ---")

    for iteration, query_string in enumerate(test_runs, start=1):
        print(f"\n[Request #{iteration}] Inbound Question: '{query_string}'")
        execution_start = time.perf_counter()

        output_payload = optimized_pipeline.invoke(query_string)

        latency = (time.perf_counter() - execution_start) * 1000
        print(f"  Latency: {latency:.2f} ms")
        print(f"  Response: {output_payload.strip()}")

    # Output total system performance statistics logs
    optimized_pipeline.tracker.log_report()
