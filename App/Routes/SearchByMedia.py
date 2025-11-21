from fastapi import APIRouter, UploadFile, File, Form
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
    from_: int|None = Form(1),
    size: int|None = Form(20)
):
    redis_service = RedisService()
    embedding = await redis_service.get_or_create_embedding(user_id, file)

    es_service = ElasticsearchService(index_name, mapping)
    result_data = es_service.search_embedding(index_name, embedding, from_=from_, size=size)
    return {
        "data": result_data,
        "message": "success"
    }
