import io
import json
import mimetypes
from typing import List, Dict, Any, Optional

import requests
from fastapi import UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from App.Services.ElasticsearchService import ElasticsearchService
from App.Helpers.GeminiEmbedding import getEmbedding, getDescriptionByAi


class MediaIngestService:
    """
    Consume 'async_medias' events, enrich with AI description from media_url,
    generate embedding from provided fields (excluding media_url), and bulk index into Elasticsearch.
    """

    def __init__(self, index_name: str = "media_embeddings_test_v3"):
        self.index_name = index_name
        self.mapping = {
            "mappings": {
                "properties": {
                    "media_id": {"type": "keyword"},
                    "name": {"type": "text"},
                    "description": {"type": "text"},
                    "ai_description": {"type": "text"},
                    "tags": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        }
        self.es_service = ElasticsearchService(self.index_name, self.mapping)

    def _download_as_uploadfile(self, url: str) -> Optional[UploadFile]:
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"⚠️ Failed to download media from URL: {url} ({resp.status_code})")
                return None
            content = resp.content
            guessed_type, _ = mimetypes.guess_type(url)
            content_type = guessed_type or resp.headers.get("Content-Type", "image/png")
            filename = url.split("/")[-1] or "image"
            file_obj = io.BytesIO(content)
            # Starlette UploadFile wraps file-like; compatible with our Gemini helpers
            upload = StarletteUploadFile(filename=filename, file=file_obj, content_type=content_type)
            return upload
        except Exception as e:
            print(f"❌ Error downloading URL {url}: {e}")
            return None

    def _build_embedding_text(self, media_name: str, description: str, ai_description: str, tag_name: str) -> str:
        parts: List[str] = []
        if media_name:
            parts.append(f"Name: {media_name}")
        if description:
            parts.append(f"Description: {description}")
        if ai_description:
            parts.append(f"AI Description: {ai_description}")
        if tag_name:
            parts.append(f"Tags: {tag_name}")
        return " \n".join(parts)

    async def _transform_event_to_doc(self, event_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            if event_obj.get("event") != "async_medias":
                return None

            media_id = str(event_obj.get("media_id")) if event_obj.get("media_id") is not None else None
            media_name = event_obj.get("media_name") or event_obj.get("name")
            media_url = event_obj.get("media_url")
            description = event_obj.get("description")
            tag_name = event_obj.get("tag_name") or event_obj.get("tags")

            ai_description = None
            if media_url:
                upload = self._download_as_uploadfile(media_url)
                if upload:
                    try:
                        ai_description = await getDescriptionByAi(upload)
                    except Exception as e:
                        print(f"⚠️ AI description generation failed for {media_url}: {e}")
                else:
                    print("⚠️ Skipping AI description; failed to build upload file")

            embed_text = self._build_embedding_text(media_name or "", description or "", ai_description or "", tag_name or "")
            embedding = await getEmbedding(text=embed_text)

            doc = {
                "media_id": media_id,
                "name": media_name,
                "description": description,
                "ai_description": ai_description,
                "tags": tag_name,
                "embedding": embedding,
            }
            return doc
        except Exception as e:
            print(f"❌ Failed to transform event to document: {e}")
            return None

    async def process_batch(self, events: List[str], chunk_size: int = 200):
        """
        Parse JSON strings, filter by event==async_medias, enrich and bulk insert.
        """
        docs: List[Dict[str, Any]] = []
        for raw in events:
            try:
                obj = json.loads(raw)
            except Exception:
                print(f"⚠️ Skip invalid JSON: {raw[:128]}...")
                continue

            doc = await self._transform_event_to_doc(obj)
            if doc:
                docs.append(doc)

        if not docs:
            print("ℹ️ No valid async_medias docs to insert in this batch")
            return

        try:
            self.es_service.insert_bulk_documents(self.index_name, docs, chunk_size=chunk_size)
        except Exception as e:
            print(f"❌ Bulk insert error: {e}")

    async def process_event(self, event: str):
        try:
            obj = json.loads(event)
        except Exception:
            print(f"⚠️ Invalid JSON event: {event[:128]}...")
            return

        doc = await self._transform_event_to_doc(obj)
        if not doc:
            return
        try:
            self.es_service.insert_bulk_documents(self.index_name, [doc], chunk_size=1)
        except Exception as e:
            print(f"❌ Insert error: {e}")
