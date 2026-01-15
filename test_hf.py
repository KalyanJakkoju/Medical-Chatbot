import os
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(repo_id, token):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation", 
        temperature=0.5,
        huggingfacehub_api_token=token,
        max_new_tokens=512
    )

try:
    llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
    response = llm.invoke("What is 2+2?")
    print("SUCCESS:", response)
except Exception as e:
    print("ERROR:", type(e).__name__, ":", e)
    import traceback
    traceback.print_exc()
