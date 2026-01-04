from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import json

from App.Core.Mysql import MySQLService
from App.Services.GeminiService import GeminiService
from App.Services.MediaIngestService import MediaIngestService

router = APIRouter(prefix="/api/v1", tags=["media"])

class MediaMetadataRequest(BaseModel):
    media_id: str


class MediaMetadataResponse(BaseModel):
    media_id: str
    title: str
    description: str
    tags: List[str]
    message: str


def _get_gemini_service() -> GeminiService:
    """Get initialized GeminiService with default config for metadata generation"""
    model = "gemini-2.5-flash-lite"
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2000
    }
    return GeminiService(
        model=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        generationConfig=generation_config
    )


@router.post("/medias/generate-metadata")
async def generate_media_metadata(req: MediaMetadataRequest):
    """
    Generate title, description, and tags for media by media_id.
    
    Processes media_url(s) which can include multiple images and videos.
    Uses LLM to generate:
    - Title: max 10 characters
    - Description: max 30 characters
    - Tags: max 10 tags as array
    """
    media_id = req.media_id

    # Query media from database
    mysql = MySQLService()
    query = """
        SELECT *
        FROM medias
        WHERE id = :media_id
        LIMIT 1
    """
    result = mysql.execute_raw_sql(query, params={"media_id": media_id})
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Media {media_id} not found")
    
    media = result[0]
    
    # Parse media_url (can be JSON string, list, or single string)
    media_url = media.get("media_url")
    if isinstance(media_url, str):
        try:
            media_url = json.loads(media_url)
        except json.JSONDecodeError:
            # If not JSON, treat as plain string
            pass

    # Get and normalize media_url(s)
    media_urls = MediaIngestService.normalize_media_urls(media_url)

    if not media_urls:
        raise HTTPException(status_code=400, detail=f"Media {media_id} has no valid media_url")

    # Process media URLs (images and videos) to get combined description
    try:
        print(f"🔄 Processing {len(media_urls)} media URL(s) for media_id: {media_id}", flush=True)
        combined_description = await MediaIngestService.process_media_urls(media_urls)
        if not combined_description:
            raise ValueError("Failed to generate description from media")
        print(f"✅ Generated description: {combined_description[:100]}...", flush=True)
    except Exception as e:
        print(f"❌ Error processing media URLs: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to process media: {str(e)}")

    # Generate metadata (title, description, tags) with strict constraints
    try:
        gemini_service = _get_gemini_service()
        metadata = await MediaIngestService.generate_metadata_with_strict_constraints(
            combined_description, gemini_service
        )
        print(f"✅ Generated metadata: {metadata}", flush=True)
    except Exception as e:
        print(f"❌ Error generating metadata: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate metadata: {str(e)}")

    title = metadata.get("title", "")
    description = metadata.get("description", "")
    tags = metadata.get("tags", [])

    # Ensure tags is a list
    if not isinstance(tags, list):
        tags = [tags] if tags else []

    return MediaMetadataResponse(
        media_id=media_id,
        title=title,
        description=description,
        tags=tags,
        message="success"
    )

