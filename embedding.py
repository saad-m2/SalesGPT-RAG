import os
import requests

# Use environment variable for API key
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY environment variable not set")

# Use the same endpoint as core.py for consistency
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/sentence-similarity"
headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"Hugging Face API error: {e}")
        print(response.text)
        raise
    return response.json()

output = query({
    "inputs": {
    "source_sentence": "That is a happy person",
    "sentences": [
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]
},
})
print(output)