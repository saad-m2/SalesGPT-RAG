from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()
import os






qdrant_client = QdrantClient(
    url="https://b1588daa-e290-4706-bc26-82576136290c.eu-central-1-0.aws.cloud.qdrant.io:6333", 
    api_key=os.getenv("QDRANT_API_KEY"),
)

print(qdrant_client.get_collections())