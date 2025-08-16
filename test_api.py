import os
from dotenv import load_dotenv
import requests

load_dotenv()

HF_API_KEY = os.getenv('HF_API_KEY')
# Use a model that's definitely available via Hugging Face inference API
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print(f"HF_API_KEY exists: {bool(HF_API_KEY)}")
print(f"HF_EMBED_MODEL: {HF_EMBED_MODEL}")

if HF_API_KEY:
    url = f'https://api-inference.huggingface.co/models/{HF_EMBED_MODEL}'
    headers = {'Authorization': f'Bearer {HF_API_KEY}'}
    
    try:
        response = requests.post(url, headers=headers, json={'inputs': 'test sentence'})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
        
        if response.status_code == 200:
            print("✅ API is working!")
        else:
            print("❌ API error")
            
    except Exception as e:
        print(f"Error: {e}")
else:
    print("❌ HF_API_KEY not found in .env file")
