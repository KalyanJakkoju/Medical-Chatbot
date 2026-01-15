import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

print(f"Testing model: {HUGGINGFACE_REPO_ID}")
client = InferenceClient(api_key=HF_TOKEN)

try:
    print("Trying text_generation...")
    response = client.text_generation("What is 2+2?", model=HUGGINGFACE_REPO_ID, max_new_tokens=10)
    print("SUCCESS (text_generation):", response)
except Exception as e:
    print("ERROR (text_generation):", e)

try:
    print("\nTrying chat_completion...")
    response = client.chat_completion(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        model=HUGGINGFACE_REPO_ID,
        max_tokens=10
    )
    print("SUCCESS (chat_completion):", response.choices[0].message.content)
except Exception as e:
    print("ERROR (chat_completion):", e)
