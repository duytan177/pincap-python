from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from App.Core.Mysql import MySQLService
from App.Services.RedisService import RedisService
from App.Services.ElasticsearchService import ElasticsearchService

router = APIRouter(prefix="/api/v1", tags=["Test"])
index_name = "media_embeddings_test_v3"
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


@router.post("/medias/search_by_image")
async def search_by_media(
    user_id: str = Form(...),
    file: UploadFile = File(None),
    from_: int|None = Form(0),
    size: int|None = Form(20)
):
    redis_service = RedisService()
    embedding = await redis_service.get_or_create_embedding(user_id, file)

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
    print(blocked_user_ids)
    must_filters = [
        {"term": {"is_deleted": False}},
    ]
    must_not_filters = []

    if blocked_user_ids:
        must_not_filters.append({"terms": {"user_id": blocked_user_ids}})

    result_data = es_service.search_embedding(index_name, embedding, must_filters, must_not_filters, from_=from_, size=size)
    return {
        "data": result_data,
        "message": "success"
    }
