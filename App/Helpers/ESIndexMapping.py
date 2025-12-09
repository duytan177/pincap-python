# ESIndexMapping.py
import os

index_name = os.getenv("ELASTIC_SEARCH_INDEX")
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
