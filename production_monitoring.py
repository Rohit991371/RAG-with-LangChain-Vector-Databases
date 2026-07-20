# User Query
#            │
#            ▼
# ┌──────────────────────┐
# │  InstrumentedLLM     │ ──> [ Starts Stopwatch Timer ]
# └──────────┬───────────┘
#            │
#            ▼
# ┌──────────────────────┐
# │  LangChain / Groq   │ ──> Fires Native Callbacks:
# │  Execution Node      │     • on_llm_start()
# └──────────┬───────────┘     • on_llm_end() ──> Captures exact token counts
#            │
#            ▼
# ┌──────────────────────┐
# │ MetricsCollector &   │ ──> Saves metrics array to memory state
# │ JSON Logging Stream  │ ──> Serializes raw metrics to stdout/file stream
# └──────────────────────┘


import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langsmith import traceable

load_dotenv()

# Configure LangSmith telemetry variables dynamically
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_PROJECT"] = os.getenv(
    "LANGSMITH_PROJECT", "production_monitoring_logging")


# 1. STRUCTURED LOGGING SUBSYSTEM
class ProductionJSONFormatter(logging.Formatter):
    """Formats Python logs into valid JSON structures for easy log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger_name": record.name,
            "log_level": record.levelname,
            "message": record.getMessage(),
            "source_code": f"{record.module}.py:{record.filename}:{record.lineno}"
        }

        # Merge extra payload attributes injected via log invocations
        if hasattr(record, "telemetry_payload"):
            log_payload.update(record.telemetry_payload)

        return json.dumps(log_payload)


def configure_structured_logger() -> logging.Logger:
    """Initializes and returns a structured JSON console output logger."""
    app_logger = logging.getLogger("ProductionRAG")
    app_logger.setLevel(logging.INFO)

    # Prevent attaching multiple duplicate handlers if re-instantiated
    if not app_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ProductionJSONFormatter())
        app_logger.addHandler(console_handler)

    return app_logger


# 2. METRICS COLLECTION CORE ENGINE
class RAGMetricsCollector:
    """Tracks, aggregates, and reports system health performance metrics."""

    def __init__(self):
        self.totals = {
            "requests_count": 0,
            "failure_count": 0,
            "accumulated_latency_ms": 0.0,
            "prompt_tokens_used": 0,
            "completion_tokens_used": 0
        }

    def record_transaction(self, latency_ms: float, prompt_tokens: int, completion_tokens: int, is_failure: bool = False):
        self.totals["requests_count"] += 1
        self.totals["accumulated_latency_ms"] += latency_ms
        self.totals["prompt_tokens_used"] += prompt_tokens
        self.totals["completion_tokens_used"] += completion_tokens
        if is_failure:
            self.totals["failure_count"] += 1

    def calculate_analytics_summary(self) -> Dict[str, Any]:
        reqs = max(self.totals["requests_count"], 1)
        total_tokens = self.totals["prompt_tokens_used"] + \
            self.totals["completion_tokens_used"]

        return {
            "total_requests_processed": self.totals["requests_count"],
            "total_exceptions_raise": self.totals["failure_count"],
            "system_error_rate": f"{(self.totals['failure_count'] / reqs) * 100:.2f}%",
            "average_response_latency_ms": round(self.totals["accumulated_latency_ms"] / reqs, 2),
            "cumulative_tokens_consumed": total_tokens,
            "average_tokens_per_payload": round(total_tokens / reqs, 2)
        }


# NATIVE LANGCHAIN METRICS CALLBACK
class TelemetryCallbackHandler(BaseCallbackHandler):
    """Listens to LangChain execution hooks to capture exact, un-estimated token values."""

    def __init__(self):
        super().__init__()
        self.last_run_prompt_tokens = 0
        self.last_run_completion_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Fires automatically when the model completes generation execution successfully."""
        self.last_run_prompt_tokens = 0
        self.last_run_completion_tokens = 0

        # Traverse generation outputs to extract exact token usage maps
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.last_run_prompt_tokens = usage.get("prompt_tokens", 0)
            self.last_run_completion_tokens = usage.get("completion_tokens", 0)


# 4. INSTRUMENTED RUNTIME LAYER
class InstrumentedProductionLLM:
    """Wrapper that wraps execution targets inside telemetry clocks, callbacks, and JSON logs."""

    def __init__(self):
        self.logger = configure_structured_logger()
        self.metrics = RAGMetricsCollector()

        # Initialize Groq via standard interface core
        self.model = init_chat_model(
            "groq:llama-3.1-8b-instant", temperature=0.0)

    @traceable(name="instrumented_pipeline_invocation")
    def execute(self, user_query: str) -> str:
        stopwatch_start = time.perf_counter()
        telemetry_handler = TelemetryCallbackHandler()

        try:
            # Inject native callback handler to capture tokens usage precisely
            response = self.model.invoke(
                user_query,
                config={"callbacks": [telemetry_handler]}
            )

            latency = (time.perf_counter() - stopwatch_start) * 1000

            # Save raw metric attributes into tracking state arrays
            self.metrics.record_transaction(
                latency_ms=latency,
                prompt_tokens=telemetry_handler.last_run_prompt_tokens,
                completion_tokens=telemetry_handler.last_run_completion_tokens,
                is_failure=False
            )

            # Stream out structural operational JSON Line logs
            self.logger.info(
                "LLM execution node transaction succeeded.",
                extra={
                    "telemetry_payload": {
                        "latency_ms": round(latency, 2),
                        "input_tokens": telemetry_handler.last_run_prompt_tokens,
                        "output_tokens": telemetry_handler.last_run_completion_tokens,
                        "status_code": 200
                    }
                }
            )
            return response.content

        except Exception as pipeline_exception:
            latency = (time.perf_counter() - stopwatch_start) * 1000

            self.metrics.record_transaction(
                latency_ms=latency, prompt_tokens=0, completion_tokens=0, is_failure=True
            )

            self.logger.error(
                "LLM execution node transaction failed.",
                extra={
                    "telemetry_payload": {
                        "latency_ms": round(latency, 2),
                        "exception_class": pipeline_exception.__class__.__name__,
                        "exception_message": str(pipeline_exception),
                        "status_code": 500
                    }
                }
            )
            raise pipeline_exception


if __name__ == "__main__":
    print("--- Starting Production Inbound Telemetry Engine ---\n")
    instrumented_agent = InstrumentedProductionLLM()

    sample_queries = [
        "What is the capital city of France?",
        "Explain multi-tenant architecture databases in ten words.",
        # Will execute fine, handled gracefully
        "Force an error value validation trigger statement."
    ]

    for iteration, query in enumerate(sample_queries, start=1):
        # In production systems, stdout catches these serialized lines for aggregation
        output = instrumented_agent.execute(query)

    print("\n" + "=" * 60)
    print("AGGREGATED ANALYTICS TELEMETRY METRICS SUMMARY REPORT:")
    print("=" * 60)
    summary_data = instrumented_agent.metrics.calculate_analytics_summary()
    for telemetry_key, telemetry_value in summary_data.items():
        print(f"  {telemetry_key:<35} : {telemetry_value}")
    print("=" * 60)
