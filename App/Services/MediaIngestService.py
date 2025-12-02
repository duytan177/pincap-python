import io
import json
import mimetypes
from typing import List, Dict, Any, Optional
import numpy as np

import requests
import asyncio
from fastapi import UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile
import os

from App.Services.CFWorkerService import CFWorkerService
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
                    "is_deleted": {"type": "boolean"},
                    "user_id": {"type": "keyword"},
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
            user_id = event_obj.get("user_id")
            existing_doc = self.es_service.get_document_by_id(self.index_name, media_id)
            if existing_doc is None:
                # Chuẩn hóa media_urls thành list
                media_urls = event_obj.get("media_url")
                if isinstance(media_urls, str):
                    media_urls = [media_urls]
                elif not isinstance(media_urls, list):
                    media_urls = []

                # ---------- PROCESS PARALLEL GET DESCRIPTION FOR IMAGE AND VIDEO---------------
                # popular extensions
                image_extensions = (
                    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
                    ".heic", ".heif", ".svg", ".ico", ".jfif", ".pjpeg", ".pjp", ".avif"
                )

                video_extensions = (
                    ".mp4", ".mov", ".wmv", ".avi", ".flv", ".mkv", ".webm", ".m4v"
                )

                # -------------------------------
                # Save by original index
                indexed_urls = [(i, url) for i, url in enumerate(media_urls) if isinstance(url, str)]
                images = [(i, url) for i, url in indexed_urls if url.lower().endswith(image_extensions)]
                videos = [(i, url) for i, url in indexed_urls if url.lower().endswith(video_extensions)]

                # -------------------------------
                # Image pipeline
                async def process_images(indexed_list):
                    uploads = await self._download_all_uploadfiles([url for i, url in indexed_list])
                    descriptions = []
                    for idx, upload in zip([i for i, _ in indexed_list], uploads):
                        try:
                            desc = await getDescriptionByAi(upload)
                            descriptions.append((idx, desc))
                        except Exception as e:
                            print(f"⚠️ AI description failed for image {upload.filename}: {e}", flush=True)
                            descriptions.append((idx, None))
                    return descriptions

                # -------------------------------
                # Video pipeline
                async def process_videos(indexed_list):
                    descriptions = []

                    for idx, url in indexed_list:
                        try:
                            print(f"video #{url}", flush=True)
                            detect_service = CFWorkerService(
                                worker_url=os.getenv("CLOUDFLARE_WORKER_PINCAP_DETECT_VIDEO")
                            )
                            description = await detect_service.extract_and_describe(url)
                            print(description)
                            descriptions.append((idx, description))
                        except Exception as e:
                            print(f"⚠️ AI description failed for video {url}: {e}", flush=True)
                            descriptions.append((idx, None))
                    return descriptions

                # -------------------------------
                # Run parallel
                image_task = asyncio.create_task(process_images(images))
                video_task = asyncio.create_task(process_videos(videos))

                image_results, video_results = await asyncio.gather(image_task, video_task)
                # -------------------------------
                # Merge and order by original media_urls
                all_results = image_results + video_results
                ai_descriptions = [desc for idx, desc in sorted(all_results, key=lambda x: x[0])]

                ai_description = " | ".join(ai_descriptions) if ai_descriptions else ""
            else:
                ai_description = existing_doc.get("ai_description")
            # Text metadata
            text_parts = [
                media_name or "",
                description or "",
                tag_name or ""
            ]
            text_content = " ".join([t for t in text_parts if t])

            # =============================
            #  call parallel 2 embedding
            # =============================

            image_embedding_task = getEmbedding(text=ai_description)
            text_embedding_task = getEmbedding(text=text_content)

            image_embedding, text_embedding = await asyncio.gather(
                image_embedding_task,
                text_embedding_task
            )

            # =============================
            #  FUSION 7 : 3
            # =============================
            embedding = fuse_embeddings(
                image_embedding,
                text_embedding,
                w_img=0.7,
                w_text=0.3
            )

            doc = {
                "media_id": media_id,
                "name": media_name,
                "description": description,
                "ai_description": ai_description,
                "tags": tag_name,
                "embedding": embedding,
                "user_id": user_id,
                "is_deleted": False
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
        print(doc, flush=True)
        if not doc:
            print(f"❌ transform event to doc error", flush=True)
            return
        try:
            self.es_service.upsert_document(
                index=self.index_name,
                id=doc["media_id"],
                document=doc
            )
        except Exception as e:
            print(f"❌ Insert error: {e}", flush=True)

def fuse_embeddings(img_emb, text_emb, w_img=0.7, w_text=0.3):
    img_emb = np.array(img_emb)
    text_emb = np.array(text_emb)

    fused = (img_emb * w_img) + (text_emb * w_text)

    # normalize L2
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm

    return fused.tolist()