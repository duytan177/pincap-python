from fastapi import APIRouter, UploadFile, File, Form
from typing import List, Optional

from pydantic import BaseModel

from App.Core.Mysql import MySQLService
from App.Helpers.GeminiEmbedding import getEmbedding
from App.Services.ElasticsearchService import ElasticsearchService
import os
router = APIRouter(prefix="/api/v1", tags=["Test"])
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
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}


# -----------------------
#  JSON Request Model
# -----------------------
class SearchTextRequest(BaseModel):
    user_id: str
    text: str
    from_: Optional[int] = 0
    size: Optional[int] = 20


@router.post("/medias/search_media_by_text")
async def search_media_by_text(req: SearchTextRequest):
    user_id = req.user_id
    text = req.text
    from_ = req.from_
    size = req.size

    embedding = await getEmbedding(text=text)
    es_service = ElasticsearchService(index_name, mapping)

    # check permission user and user blocked then not search
    mysql = MySQLService()
    query = """
        SELECT followee_id AS blocked_user_id
        FROM user_relationship
        WHERE follower_id = :user_id
          AND user_status = :user_status
    """
    blocked_rows = mysql.execute_raw_sql(query, params={"user_id": user_id, "user_status": "0"})

    blocked_user_ids: List[str] = [r["blocked_user_id"] for r in blocked_rows]
    must_filters = [
        {"term": {"is_deleted": False}},
    ]
    must_not_filters = []

    if blocked_user_ids:
        must_not_filters.append({"terms": {"user_id": blocked_user_ids}})

    result_data = es_service.search_embedding(index_name, embedding, must_filters, must_not_filters,min_score=0.79, from_=from_, size=size)

    formatted = es_service.format_media_ids(result_data, from_=from_, size=size)

    return {
        "media_ids": formatted["media_ids"],
        "has_more": formatted["has_more"],
        "total": formatted["total"],
        "message": "success"
    }

