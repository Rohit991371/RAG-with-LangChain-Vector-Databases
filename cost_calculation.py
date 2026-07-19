import os
from typing import Tuple, Dict, Any
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

load_dotenv()


class TokenBudgetGuardrail:
    """Tracks exact Groq token metrics and enforces constraints on inbound prompts."""

    def __init__(self, max_input_tokens: int = 150):
        self.max_input_tokens = max_input_tokens
        self.stats = {"total_input_tokens": 0,
                      "total_output_tokens": 0, "successful_requests": 0}

    def estimate_tokens(self, text: str) -> int:
        """
        Heuristic pre-check estimation baseline.
        Using a standard rule of thumb: 1 word ≈ 1.33 tokens.
        """
        return int(len(text.split()) * 1.33)

    def check_preflight_budget(self, text: str) -> Tuple[bool, int]:
        """Validates if the prompt looks safely within boundaries before executing network requests."""
        estimated_tokens = self.estimate_tokens(text)
        return estimated_tokens <= self.max_input_tokens, estimated_tokens

    def record_actual_usage(self, response_message: AIMessage):
        """Extracts the exact raw token tracking metrics returned by Groq's execution payload."""
        metadata = response_message.response_metadata
        token_usage = metadata.get("token_usage", {})

        # Extract real token fields from Groq API response dict
        input_tokens = token_usage.get("prompt_tokens", 0)
        output_tokens = token_usage.get("completion_tokens", 0)

        self.stats["total_input_tokens"] += input_tokens
        self.stats["total_output_tokens"] += output_tokens
        self.stats["successful_requests"] += 1

    def get_summary_report(self) -> Dict[str, Any]:
        """Generates analysis reports for consumption monitoring logs."""
        total = self.stats["total_input_tokens"] + \
            self.stats["total_output_tokens"]
        return {
            **self.stats,
            "cumulative_tokens_used": total,
            "avg_tokens_per_call": round(total / max(self.stats["successful_requests"], 1), 2)
        }


class BudgetedGroqLLM:
    """Wrapper that applies token constraints and metrics tracking to Groq models."""

    def __init__(self, max_input_tokens: int = 150):
        # Initialize Groq via the standard integration routing core
        self.llm = init_chat_model(
            "groq:llama-3.3-70b-versatile", temperature=0.0)
        self.guardrail = TokenBudgetGuardrail(
            max_input_tokens=max_input_tokens)

    def invoke(self, query: str) -> str:
        # 1. Perform pre-flight check to block bloated inputs early
        is_safe, est_tokens = self.guardrail.check_preflight_budget(query)
        if not is_safe:
            raise ValueError(
                f"Blocked: Query estimation ({est_tokens} tokens) exceeds budget cap ({self.guardrail.max_input_tokens})."
            )

        # 2. Fire Request
        response = self.llm.invoke(query)

        # 3. Log accurate usage statistics from the real provider response
        self.guardrail.record_actual_usage(response)

        return response.content


# --- RUN CONTEXT TESTS ---
if __name__ == "__main__":
    # Initialize with a tight budget restriction of 50 tokens for testing
    budgeted_agent = BudgetedGroqLLM(max_input_tokens=50)

    test_queries = [
        # Short prompt -> Passes pre-flight check
        "Explain machine learning in one short sentence.",
        "Provide a massive essay explaining " + "nested loops " *
        60 + "in high detail."  # Spammed text -> Blocked early
    ]

    print("\n--- Starting Cost Optimization Demo ---")

    for q in test_queries:
        print(f"\nEvaluating: '{q[:60]}...'")
        try:
            answer = budgeted_agent.invoke(q)
            print(f"  Result Status: ✅ SUCCESS")
            print(f"  LLM Response: {answer.strip()}")
        except ValueError as budget_exception:
            print(f"  Result Status: ❌ SHIELD TRIGGERED")
            print(f"  Reason: {budget_exception}")

    print("\n" + "=" * 50)
    print("FINAL ACCURATE CONSUMPTION REPORT:")
    print("=" * 50)
    for key, val in budgeted_agent.guardrail.get_summary_report().items():
        print(f"  {key:<25} : {val}")
