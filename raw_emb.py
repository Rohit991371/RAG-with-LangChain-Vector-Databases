from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

import os

load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# CHAT MODEL
# conversation = client.chat.completions.create(
#     model = "openai/gpt-oss-safeguard-20b",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the capital of France?"},
#     ],
# )

# print(conversation.choices[0].message.content)


client = InferenceClient(api_key=os.getenv("HF_TOKEN"))

# EMBEDDING MODEL
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but we can
# also pass your own prompt via the `prompt` argument
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)