# core.py
import os
import requests
import pandas as pd
import numpy as np
import json
import time
from typing import List, Dict
from pathlib import Path
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from rank_bm25 import BM25Okapi

# ----------------------------
# Environment & API Settings
# ----------------------------
HF_API_KEY = os.getenv("HF_API_KEY")
# Use the router API with feature-extraction pipeline for embeddings
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_EMBED_MODEL}/pipeline/feature-extraction"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_URL")  # Use QDRANT_URL from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ----------------------------
# Qdrant Client
# ----------------------------
qdrant_client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

# ----------------------------
# Hugging Face Embeddings
# ----------------------------
def hf_embed_batch(texts: List[str], batch_size: int = 8) -> List[List[float]]:
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Use the standard inference API format
        payload = {"inputs": batch}
        
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            print(f"Hugging Face API error: {e}")
            print(response.text)
            raise

        result = response.json()
        
        if isinstance(result, dict) and "error" in result:
            raise ValueError(f"Hugging Face API error: {result['error']}")
        
        # The feature-extraction pipeline returns embeddings directly
        if isinstance(result, list):
            all_embeddings.extend(result)
        else:
            raise ValueError(f"Unexpected response format: {result}")

    return all_embeddings

# ----------------------------
# PDF/CSV Loading
# ----------------------------
def load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_csv_text(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    return df.to_string(index=False)

# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ----------------------------
# Vector / Qdrant Functions
# ----------------------------
def create_collection_if_not_exists(collection_name: str, vector_size: int):
    try:
        qdrant_client.get_collection(collection_name=collection_name)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance="Cosine")
        )

def insert_vectors(collection_name: str, texts: List[str], metadatas: List[Dict]):
    embeddings = hf_embed_batch(texts)
    points = []
    for idx, (vec, meta) in enumerate(zip(embeddings, metadatas)):
        points.append(PointStruct(id=idx, vector=vec, payload=meta))
    create_collection_if_not_exists(collection_name, vector_size=len(embeddings[0]))
    qdrant_client.upsert(collection_name=collection_name, points=points)

# ----------------------------
# Normalization
# ----------------------------
def l2_normalize(vectors: List[List[float]]) -> List[List[float]]:
    normalized = []
    for vec in vectors:
        arr = np.array(vec, dtype=float)
        norm = np.linalg.norm(arr)
        if norm == 0:
            normalized.append(vec)
        else:
            normalized.append((arr / norm).tolist())
    return normalized

# ----------------------------
# BM25
# ----------------------------
def build_bm25(records: List[Dict]) -> str:
    """Build BM25 index and save to file"""
    texts = [r["text"] for r in records]
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    
    # Save BM25 model
    import pickle
    bm25_path = "data/bm25_model.pkl"
    os.makedirs("data", exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    
    return bm25_path

# ----------------------------
# Hybrid Search
# ----------------------------
def hybrid_search(query: str, top: int = 8, collection_name: str = None) -> List[Dict]:
    if collection_name is None:
        collection_name = os.getenv("QDRANT_COLLECTION", "sales_docs")
    """Perform hybrid search combining dense and sparse retrieval"""
    
    # Dense search (vector similarity)
    query_embedding = hf_embed_batch([query])[0]
    query_embedding = l2_normalize([query_embedding])[0]
    
    dense_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top * 2  # Get more for reranking
    )
    
    # Sparse search (BM25)
    try:
        import pickle
        with open("data/bm25_model.pkl", "rb") as f:
            bm25 = pickle.load(f)
        
        # Get all documents for BM25 search
        all_docs = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000
        )[0]
        
        texts = [doc.payload.get("text", "") for doc in all_docs]
        tokenized = [t.split() for t in texts]
        bm25_scores = bm25.get_scores(query.split())
        
        # Combine results (simple approach)
        combined_results = []
        for i, doc in enumerate(all_docs):
            dense_score = 0
            for dense_result in dense_results:
                if dense_result.id == doc.id:
                    dense_score = dense_result.score
                    break
            
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0
            
            # Simple weighted combination
            combined_score = 0.7 * dense_score + 0.3 * bm25_score
            
            combined_results.append({
                "id": doc.id,
                "text": doc.payload.get("text", ""),
                "score": combined_score,
                "doc_id": doc.payload.get("source", "unknown"),
                "chunk_id": doc.id
            })
        
        # Sort by combined score and return top results
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:top]
        
    except FileNotFoundError:
        # Fallback to dense search only
        return [
            {
                "id": result.id,
                "text": result.payload.get("text", ""),
                "score": result.score,
                "doc_id": result.payload.get("source", "unknown"),
                "chunk_id": result.id
            }
            for result in dense_results[:top]
        ]

# ----------------------------
# LLM Integration (Groq)
# ----------------------------
def call_groq_chat(prompt: str, model: str = "llama3-8b-8192") -> tuple:
    """Call Groq API for chat completion"""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    took = time.time() - start_time
    
    if response.status_code != 200:
        raise Exception(f"Groq API error: {response.status_code} - {response.text}")
    
    return response.json(), took

# ----------------------------
# Prompt Templates
# ----------------------------
def pitch_prompt(company: str, industry: str, goal: str, snippets: List[Dict], focus: str = None) -> str:
    """Generate a prompt for pitch generation"""
    
    context = "\n\n".join([f"Source {i+1}: {s['text']}" for i, s in enumerate(snippets[:5])])
    
    prompt = f"""You are an expert B2B sales professional. Generate a {goal} for {company} in the {industry} industry.

Company: {company}
Industry: {industry}
Goal: {goal}
{f"Focus Area: {focus}" if focus else ""}

Use the following source materials to create a compelling, personalized pitch:

{context}

Instructions:
1. Make it specific to {company} and {industry}
2. Include relevant examples from the source materials
3. Keep it professional but engaging
4. Structure it clearly with proper formatting
5. Include a clear call-to-action

Generate the {goal}:"""

    return prompt

# ----------------------------
# Evaluation
# ----------------------------
def recall_at_k(eval_file: str, k: int = 10, collection_name: str = None) -> float:
    if collection_name is None:
        collection_name = os.getenv("QDRANT_COLLECTION", "sales_docs")
    """Compute Recall@K for evaluation dataset"""
    
    df = pd.read_csv(eval_file)
    total_queries = len(df)
    correct_retrievals = 0
    
    for _, row in df.iterrows():
        query = row['query']
        relevant_docs = row['relevant_docs'].split('|') if '|' in str(row['relevant_docs']) else [row['relevant_docs']]
        
        # Get top-k results
        results = hybrid_search(query, top=k, collection_name=collection_name)
        retrieved_docs = [r['doc_id'] for r in results]
        
        # Check if any relevant doc is in top-k
        if any(doc in retrieved_docs for doc in relevant_docs):
            correct_retrievals += 1
    
    return correct_retrievals / total_queries if total_queries > 0 else 0.0

# ----------------------------
# Wrapper Functions for App
# ----------------------------
def ensure_collection(vector_size: int, collection_name: str = None):
    if collection_name is None:
        collection_name = os.getenv("QDRANT_COLLECTION", "sales_docs")
    """Ensure collection exists with given vector size"""
    create_collection_if_not_exists(collection_name, vector_size)

def ingest_pdf(pdf_path: str, meta: dict = None) -> List[Dict]:
    """Ingest PDF and return records for processing"""
    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)
    
    records = []
    for i, chunk in enumerate(chunks):
        record = {
            "text": chunk,
            "source": pdf_path,
            "chunk_id": i,
            **(meta or {})
        }
        records.append(record)
    
    return records

def upsert_records(records: List[Dict], vectors: List[List[float]], collection_name: str = None):
    if collection_name is None:
        collection_name = os.getenv("QDRANT_COLLECTION", "sales_docs")
    """Upsert records with vectors to Qdrant"""
    points = []
    for i, (record, vec) in enumerate(zip(records, vectors)):
        point = PointStruct(
            id=i,
            vector=vec,
            payload=record
        )
        points.append(point)
    
    qdrant_client.upsert(collection_name=collection_name, points=points)
