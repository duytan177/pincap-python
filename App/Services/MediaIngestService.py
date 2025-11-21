import io
import json
import mimetypes
from typing import List, Dict, Any, Optional

import requests
import asyncio
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
                    "media_url": {"type": "keyword"},
                }
            }
        }
        self.es_service = ElasticsearchService(self.index_name, self.mapping)

    def _download_as_uploadfile(self, url: str) -> Optional[UploadFile]:
        """Download media URL and convert to UploadFile in memory"""
        try:
            print(f"Get file {url}")
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"⚠️ Failed to download media from URL: {url} ({resp.status_code})")
                return None
            content = resp.content
            guessed_type, _ = mimetypes.guess_type(url)
            filename = url.split("/")[-1] or "file"
            file_obj = io.BytesIO(content)
            upload = StarletteUploadFile(filename=filename, file=file_obj)
            return upload
        except Exception as e:
            print(f"❌ Error downloading URL {url}: {e}")
            return None

    async def _download_all_uploadfiles(self, urls: List[str], max_concurrent=3) -> List[UploadFile]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def sem_download(url):
            async with semaphore:
                return await asyncio.to_thread(self._download_as_uploadfile, url)

        tasks = [sem_download(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    def _build_embedding_text(self, media_name: str, description: str, ai_description: str, tag_name: str) -> str:
        parts: List[str] = []
        if media_name:
            parts.append(f"Name: {media_name}")
        if description:
            parts.append(f"Description: {description}")
        if ai_description:
            parts.append(f"AI Description: {ai_description}")
        if tag_name:
            # Nếu tag_name là list hoặc string
            if isinstance(tag_name, list):
                parts.append(f"Tags: {', '.join(tag_name)}")
            else:
                parts.append(f"Tags: {tag_name}")
        return " \n".join(parts)

    async def _transform_event_to_doc(self, event_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            media_id = str(event_obj.get("media_id")) if event_obj.get("media_id") else None
            media_name = event_obj.get("media_name") or event_obj.get("name")
            description = event_obj.get("description")
            tag_name = event_obj.get("tag_name") or event_obj.get("tags")
            # Chuẩn hóa media_urls thành list
            media_urls = event_obj.get("media_url")
            if isinstance(media_urls, str):
                media_urls = [media_urls]
            elif not isinstance(media_urls, list):
                media_urls = []

            # Download tất cả media files
            uploads = await self._download_all_uploadfiles(media_urls)

            # # Lấy AI description cho tất cả media
            ai_descriptions = []
            for upload in uploads:
                try:
                    desc = await getDescriptionByAi(upload)
                    ai_descriptions.append(desc)
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"⚠️ AI description generation failed for {upload.filename}: {e}", flush=True)

            ai_description = " | ".join(ai_descriptions) if ai_descriptions else ""

            # Build embedding text
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
            print(f"❌ Failed to transform event to document: {e}", flush=True)
            return None

    # def process_batch(self, events: List[str], chunk_size: int = 200):
    #     """Parse JSON strings, enrich, and bulk insert into Elasticsearch"""
    #     docs: List[Dict[str, Any]] = []
    #     for raw in events:
    #         try:
    #             print(f"row json event: {raw}")
    #             obj = json.loads(raw)
    #         except Exception:
    #             print(f"⚠️ Skip invalid JSON: {raw[:128]}...")
    #             continue
    #
    #         # doc = self._transform_event_to_doc(obj)
    #         # if doc:
    #         #     docs.append(doc)
    #
    #     if not docs:
    #         print("ℹ️ No valid async_medias docs to insert in this batch")
    #         return
    #     #
    #     # try:
    #     #     self.es_service.insert_bulk_documents(self.index_name, docs, chunk_size=chunk_size)
    #     # except Exception as e:
    #     #     print(f"❌ Bulk insert error: {e}")

    async def process_event(self, event: dict):
        """Process a single event"""
        doc = await self._transform_event_to_doc(event)
        if not doc:
            print(f"❌ transform event to doc error", flush=True)
            return
        try:
            self.es_service.insert_document(self.index_name, doc["media_id"], doc)
        except Exception as e:
            print(f"❌ Insert error: {e}", flush=True)
