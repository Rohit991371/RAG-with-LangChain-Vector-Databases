import os
from dotenv import load_dotenv


from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

load_dotenv()

# Turning tracing to "true" forces LangChain hooks to stream telemetry automatically
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv(
    "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

os.environ["LANGSMITH_PROJECT"] = os.getenv(
    "LANGSMITH_PROJECT", "rag_production_monitoring")

print("Initializing unified ChatGroq engine...")
llm = init_chat_model("groq:llama-3.3-70b-versatile", temperature=0.0)


@traceable(name="basic_chain_execution")
def basic_tracing():
    """Demonstrates how automatic LCEL routing pipes logging traces to LangSmith."""
    prompt = ChatPromptTemplate.from_template("Explain the concept of {topic} in one short sentence.")
    chain = prompt | llm | StrOutputParser()
    
    print("\n=== Running Baseline LCEL Automatic Tracing ===")
    result = chain.invoke({"topic": "Retrieval-Augmented Generation"})
    print(f"Result: {result.strip()}")


@traceable(name="production_summarize_workflow", tags=["pipeline_v1", "summarization"])
def named_runs():
    """Demonstrates appending tag constraints to track down specific dashboard logs."""
    prompt = ChatPromptTemplate.from_template("Provide a bulleted summary for the text: {text}")
    chain = prompt | llm | StrOutputParser()
    
    print("\n=== Running Categorized Tagged Operations ===")
    sample_corpus = "LangSmith provides complete lifecycle DevOps visibility, tracking metrics from evaluation to production logs."
    result = chain.invoke({"text": sample_corpus})
    print(f"Result:\n{result.strip()}")
    
    
@traceable(name="authenticated_user_request", tags=["user_inbound"])
def trace_with_metadata(user_id: str, client_tier: str, question: str) -> str:
    """
    Demonstrates injecting request-level context. 
    LangSmith automatically extracts keyword arguments passed to functions 
    decorated with @traceable and registers them as searchable run metadata attributes.
    """
    
    prompt = ChatPromptTemplate.from_template("Answer the user request: {question}")
    chain = prompt | llm | StrOutputParser()

    print(f"\n=== Processing Request for {user_id} [Tier: {client_tier}] ===")
    # Pass input variables - metadata filters are derived directly from the inputs
    return chain.invoke({"question": question})



if __name__ == "__main__":
    # Ensure your GROQ_API_KEY is populated before execution!
    if not os.getenv("GROQ_API_KEY"):
        print("CRITICAL ERROR: Please define your GROQ_API_KEY inside your local environment setup.")
    else:
        basic_tracing()
        named_runs()
        
        # Simulating distinct active client requests passing custom attributes
        response = trace_with_metadata(
            user_id="enterprise_client_88", 
            client_tier="Premium_Tier", 
            question="What is an LLM context window?"
        )
        print(f"Server Response: {response.strip()}")
        
        print("\n" + "="*70)
        print("Telemetry execution pipeline finished.")
        print("Navigate to https://smith.langchain.com to monitor exact run graphs.")
        print("="*70)