import json
import io
import mimetypes
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import os
import requests
from fastapi import UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from App.Services.ElasticsearchService import ElasticsearchService
from App.Services.CFWorkerService import CFWorkerService
from App.Services.GeminiService import GeminiService
from App.Helpers.GeminiEmbedding import getEmbedding, getDescriptionByAi

# Media file extensions
IMAGE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
    ".heic", ".heif", ".svg", ".ico", ".jfif", ".pjpeg", ".pjp", ".avif"
)

VIDEO_EXTENSIONS = (
    ".mp4", ".mov", ".wmv", ".avi", ".flv", ".mkv", ".webm", ".m4v"
)


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

    # =============================
    # REUSABLE STATIC METHODS
    # =============================

    @staticmethod
    def normalize_media_urls(media_url: str | List[str] | None) -> List[str]:
        """Normalize media_url to a list of strings"""
        if not media_url:
            return []
        if isinstance(media_url, str):
            return [media_url]
        elif isinstance(media_url, list):
            return [url for url in media_url if isinstance(url, str)]
        return []

    @staticmethod
    def download_as_uploadfile(url: str) -> Optional[UploadFile]:
        """Download media URL and convert to UploadFile in memory"""
        try:
            print(f"📥 Downloading media from URL: {url}", flush=True)
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"⚠️ Failed to download media from URL: {url} ({resp.status_code})", flush=True)
                return None
            content = resp.content
            guessed_type, _ = mimetypes.guess_type(url)
            filename = url.split("/")[-1] or "file"
            file_obj = io.BytesIO(content)
            upload = StarletteUploadFile(filename=filename, file=file_obj)
            return upload
        except Exception as e:
            print(f"❌ Error downloading URL {url}: {e}", flush=True)
            return None

    @staticmethod
    async def download_all_uploadfiles(urls: List[str], max_concurrent: int = 3) -> List[UploadFile]:
        """Download multiple URLs concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def sem_download(url):
            async with semaphore:
                return await asyncio.to_thread(MediaIngestService.download_as_uploadfile, url)

        tasks = [sem_download(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    @staticmethod
    def categorize_media_urls(media_urls: List[str]) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
        """
        Categorize media URLs into images and videos.
        Returns: (images, videos) where each is a list of (index, url) tuples
        """
        indexed_urls = [(i, url) for i, url in enumerate(media_urls) if isinstance(url, str)]
        images = [(i, url) for i, url in indexed_urls if url.lower().endswith(IMAGE_EXTENSIONS)]
        videos = [(i, url) for i, url in indexed_urls if url.lower().endswith(VIDEO_EXTENSIONS)]
        return images, videos

    @staticmethod
    async def process_images(indexed_images: List[Tuple[int, str]]) -> List[Tuple[int, Optional[str]]]:
        """
        Process images and generate descriptions.
        Returns: List of (index, description) tuples
        """
        if not indexed_images:
            return []

        uploads = await MediaIngestService.download_all_uploadfiles(
            [url for i, url in indexed_images]
        )
        descriptions = []
        for idx, upload in zip([i for i, _ in indexed_images], uploads):
            try:
                desc = await getDescriptionByAi(upload)
                descriptions.append((idx, desc))
                await time.sleep(0.2)
            except Exception as e:
                print(f"⚠️ AI description failed for image {upload.filename}: {e}", flush=True)
                descriptions.append((idx, None))
        return descriptions

    @staticmethod
    async def process_videos(
        indexed_videos: List[Tuple[int, str]],
        worker_url: Optional[str] = None
    ) -> List[Tuple[int, Optional[str]]]:
        """
        Process videos and generate descriptions.
        Returns: List of (index, description) tuples
        """
        if not indexed_videos:
            return []

        descriptions = []
        worker_url = worker_url or os.getenv("CLOUDFLARE_WORKER_PINCAP_DETECT_VIDEO", "")
        detect_service = CFWorkerService(worker_url=worker_url)

        for idx, url in indexed_videos:
            try:
                print(f"🎥 Processing video #{url}", flush=True)
                description = await detect_service.extract_and_describe(url)
                if isinstance(description, dict) and "error" in description:
                    print(f"⚠️ Video processing error: {description['error']}", flush=True)
                    descriptions.append((idx, None))
                else:
                    descriptions.append((idx, description))
            except Exception as e:
                print(f"⚠️ AI description failed for video {url}: {e}", flush=True)
                descriptions.append((idx, None))
        return descriptions

    @staticmethod
    async def process_media_urls(
        media_urls: List[str],
        worker_url: Optional[str] = None
    ) -> str:
        """
        Process all media URLs (images and videos) in parallel and return combined description.
        
        Args:
            media_urls: List of media URLs to process
            worker_url: Optional Cloudflare worker URL for video processing
            
        Returns:
            Combined description string from all media
        """
        # Categorize media
        images, videos = MediaIngestService.categorize_media_urls(media_urls)

        # Process in parallel
        image_task = asyncio.create_task(
            MediaIngestService.process_images(images)
        )
        video_task = asyncio.create_task(
            MediaIngestService.process_videos(videos, worker_url)
        )

        image_results, video_results = await asyncio.gather(image_task, video_task)

        # Merge and order by original media_urls
        all_results = image_results + video_results
        descriptions = [
            desc for idx, desc in sorted(all_results, key=lambda x: x[0])
            if desc is not None
        ]

        # Combine descriptions
        combined_description = " | ".join(descriptions) if descriptions else ""
        return combined_description

    @staticmethod
    async def generate_metadata_from_description(
        combined_description: str,
        gemini_service: GeminiService
    ) -> Dict[str, Any]:
        """
        Generate title, description, and tags from combined media description using Gemini AI.
        
        Args:
            combined_description: Combined description from all media
            gemini_service: Initialized GeminiService instance
            
        Returns:
            Dictionary with 'title', 'description', and 'tags' keys
        """
        if not combined_description:
            raise ValueError("❌ Combined description cannot be empty")

        # Generate title, description, and tags using Gemini
        system_prompt = """
        You are a content analysis assistant.
        
        ## TASK
        Analyze the provided media description and generate:
        1. A concise, engaging title (max 10 words)
        2. A detailed description (2-3 sentences)
        3. Relevant tags (max 10 keywords, comma-separated)
        
        ## OUTPUT FORMAT
        Return ONLY a valid JSON object with this exact structure:
        {
            "title": "string",
            "description": "string",
            "tags": ["tag1", "tag2", "tag3"]
        }
        
        ## RULES
        - Title should be catchy and descriptive
        - Description should be informative and detailed
        - Tags should be relevant keywords (lowercase, no spaces in tags)
        - Return ONLY the JSON, no additional text
        """

        user_prompt = f"""
        Based on this media content description, generate a title, description, and tags:
        
        {combined_description}
        
        Return the JSON object as specified.
        """

        # Build prompt
        prompt = gemini_service.buildPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            files=None
        )
        print(prompt, flush=True)
        # Call Gemini API
        response_text = await gemini_service.textToText(prompt)

        # Parse JSON response
        try:
            # Try to extract JSON from response (in case there's extra text)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                metadata = json.loads(json_str)
            else:
                # If no JSON object is found, trigger the exception to use fallback
                raise json.JSONDecodeError("No JSON object found in response", response_text, 0)
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON response: {response_text}. Error: {e}", flush=True)
            # Fallback: create basic metadata
            words = combined_description.split()
            metadata = {
                "title": " ".join(words[:10]),
                "description": combined_description[:200],
                "tags": words[:5] if len(words) >= 5 else words
            }

        return metadata

    # =============================
    # INSTANCE METHODS
    # =============================

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
                # Normalize media_urls to list using reusable method
                media_url = event_obj.get("media_url")
                media_urls = MediaIngestService.normalize_media_urls(media_url)

                # Process media URLs using reusable method
                if media_urls:
                    ai_description = await MediaIngestService.process_media_urls(media_urls)
                else:
                    ai_description = ""
            else:
                ai_description = existing_doc.get("ai_description")
            # Text metadata
            text_parts = [
                media_name or "",
                description or "",
                tag_name or ""
            ]
            text_content = " ".join([t for t in text_parts if t]).strip()

            # =============================
            #   Build embedding tasks
            # =============================

            tasks = []

            # image embedding
            tasks.append(getEmbedding(text=ai_description))

            # text embedding (only run if text_content != "")
            # if text_content:
            #     tasks.append(getEmbedding(text=text_content))
            # =============================
            #   Run tasks parallel
            # =============================
            results = await asyncio.gather(*tasks)

            image_embedding = results[0]
            text_embedding = results[1] if len(results) > 1 else None


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

def fuse_embeddings(img_emb, text_emb, w_img=0.5, w_text=0.5):
    """
    Fuse 2 embeddings an toàn, hỗ trợ text_emb = None.
    Nếu text_emb = None → trả về img_emb (L2 normalized).
    """

    img_emb = np.array(img_emb, dtype=float)

    # Nếu không có text embedding → dùng toàn bộ img_emb
    if text_emb is None:
        fused = img_emb
    else:
        text_emb = np.array(text_emb, dtype=float)
        fused = (img_emb * w_img) + (text_emb * w_text)

    # Normalize
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm

    return fused.tolist()