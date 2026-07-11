from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")

from langchain_groq import ChatGroq


print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():
    llm_groq = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0)
    response = llm_groq.invoke("Say 'Setup complete! in one word'")
    print(f"Response from ChatGroq: {response}")
    
    print("setup complete")


if __name__ == "__main__":
    main()
