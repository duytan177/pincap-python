# ESIndexMapping.py
import os

index_name = os.getenv("ELASTIC_SEARCH_INDEX")
if not index_name:
    raise RuntimeError("ELASTIC_SEARCH_INDEX environment variable is not set or empty.")

mapping = {
    "mappings": {
        "properties": {
            "media_id": {"type": "keyword"},
            "name": {"type": "text"},
            "description": {"type": "text"},
            "ai_description": {"type": "text"},
            "tags": {"type": "keyword"},
            "is_deleted": {"type": "boolean"},
            "media_url": {"type": "keyword"},
            "user_id": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

index_user = os.getenv("USER_EMBEDDING_INDEX")

user_embedding_mapping = {
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},  # Unique identifier for the user
            "updated_at": {"type": "date"},
            "embedding": {  # The embedding vector for the user
                "type": "dense_vector",
                "dims": 1536,  # Number of dimensions for the user embedding
                "index": True,  # Indexing the vector for similarity search
                "similarity": "cosine"  # Cosine similarity for comparing user vectors
            }
        }
    }
}

WEIGHTS_FOR_USER_EMBEDDING = {
    "view": 0.1,
    "like": 0.2,
    "comment": 0.3,
    "save": 0.3,
    "search": 0.2
}