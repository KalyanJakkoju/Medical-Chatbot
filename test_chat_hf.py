import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

print(f"Testing ChatHuggingFace with model: {repo_id}")

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=HF_TOKEN,
    # task="text-generation" # Let's see if it works without task
)
chat_model = ChatHuggingFace(llm=llm)

try:
    response = chat_model.invoke([HumanMessage(content="What is 2+2?")])
    print("SUCCESS:", response.content)
except Exception as e:
    print("ERROR:", e)
