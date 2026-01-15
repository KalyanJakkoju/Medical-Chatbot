import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
MODELS = [
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-2-9b-it"
]

client = InferenceClient(api_key=HF_TOKEN)

for model in MODELS:
    print(f"\n--- Testing model: {model} ---")
    try:
        response = client.text_generation("What is 2+2?", model=model, max_new_tokens=10)
        print(f"SUCCESS (text_generation): {response.strip()}")
    except Exception as e:
        print(f"ERROR (text_generation): {e}")

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            model=model,
            max_tokens=10
        )
        print(f"SUCCESS (chat_completion): {response.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"ERROR (chat_completion): {e}")
