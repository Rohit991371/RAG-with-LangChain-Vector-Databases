from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import InferenceClient
import os

load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

## CHAT MODEL
# conversation = client.chat.completions.create(
#     model = "openai/gpt-oss-safeguard-20b",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the capital of France?"},
#     ],
# )

# print(conversation.choices[0].message.content)


client = InferenceClient(api_key=os.getenv("HF_TOKEN"))

##  EMBEDDING MODEL
response = client.embeddings.create(
    input = "Your text string goes here",
    model="Qwen/Qwen3-Embedding-0.6B"
)

print(response)
print(len(response.data[0].embedding))
