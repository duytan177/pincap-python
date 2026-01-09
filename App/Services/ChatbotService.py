import json
import re
import os
import io
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from App.Services.GeminiService import GeminiService
from App.Services.ElasticsearchService import ElasticsearchService
from App.Helpers.GeminiEmbedding import getEmbedding
from App.Helpers.ESIndexMapping import index_name, mapping
from App.Core.Mysql import MySQLService


class ChatbotService:
    """
    Chatbot service using Gemini Flash 2.5 + RAG for media management.
    Handles intent detection, RAG retrieval, and structured responses.
    """

    def __init__(self):
        self.llm_model = "gemini-2.5-flash"
        self.llm_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.llm_model}:generateContent"
        
        # LLM configuration
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8048,
        }
        
        self.gemini_service = GeminiService(
            model=self.llm_url,
            generationConfig=self.generation_config
        )
        
        self.es_service = ElasticsearchService(index_name, mapping)

    async def detect_intent(self, user_message: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Detect user intent from message.
        Returns: SEARCH_MEDIA, SUGGEST_MEDIA, CONFIRM_CREATE_ALBUM, CREATE_MEDIA_FROM_INPUT, or GENERAL_QA
        """
        system_prompt = """
        You are an intent detection system for a media management chatbot.
        
        ## TASK
        Analyze the user's message and determine the intent.
        
        ## INTENTS
        1. SEARCH_MEDIA - User asks to search, list, or find media (e.g., "Liệt kê 10 media phổ biến nhất", "Tìm media về anime")
        2. SUGGEST_MEDIA - User asks for suggestions or recommendations (e.g., "Gợi ý cho tôi 20 media chủ đề anime One Piece")
        3. CONFIRM_CREATE_ALBUM - User confirms creating an album (e.g., "Có", "Đồng ý", "Tạo album", "OK")
        4. CREATE_MEDIA_FROM_INPUT - User provides a file URL or wants to create media from input (e.g., "Tạo media từ URL này: ...", "Thêm media từ file")
        5. GENERAL_QA - General questions that don't fit above categories
        
        ## OUTPUT
        Return ONLY the intent name (e.g., SEARCH_MEDIA) - no additional text.
        """
        
        user_prompt = f"User message: {user_message}\n\nDetect the intent:"
        
        history = conversation_history or []
        prompt = self.gemini_service.buildPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history
        )
        
        # Debug log
        print(f"\n🔍 [INTENT DETECTION] Prompt:", flush=True)
        print(json.dumps(prompt, indent=2, ensure_ascii=False), flush=True)
        
        response = await self.gemini_service.textToText(prompt)
        intent = response.strip().upper()
        
        print(f"🔍 [INTENT DETECTION] Response: {intent}", flush=True)
        
        # Validate intent
        valid_intents = ["SEARCH_MEDIA", "SUGGEST_MEDIA", "CONFIRM_CREATE_ALBUM", "CREATE_MEDIA_FROM_INPUT", "GENERAL_QA"]
        if intent not in valid_intents:
            # Fallback: try to infer from keywords
            message_lower = user_message.lower()
            if any(word in message_lower for word in ["tìm", "liệt kê", "danh sách", "search", "find"]):
                intent = "SEARCH_MEDIA"
            elif any(word in message_lower for word in ["gợi ý", "suggest", "recommend", "đề xuất"]):
                intent = "SUGGEST_MEDIA"
            elif any(word in message_lower for word in ["có", "đồng ý", "ok", "tạo album", "yes"]):
                intent = "CONFIRM_CREATE_ALBUM"
            elif any(word in message_lower for word in ["url", "file", "tạo media", "thêm media", "create media"]):
                intent = "CREATE_MEDIA_FROM_INPUT"
            else:
                intent = "GENERAL_QA"
        
        return intent

    async def retrieve_media_rag(
        self,
        query: str,
        user_id: str,
        limit: int = 20,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant media from Elasticsearch using RAG.
        Returns list of media documents with metadata.
        """
        # Generate embedding for query
        query_embedding = await getEmbedding(text=query)
        
        # Get blocked users
        mysql = MySQLService()
        query_sql = """
            SELECT followee_id AS blocked_user_id
            FROM user_relationship
            WHERE follower_id = :user_id
              AND user_status = :user_status
        """
        blocked_rows = mysql.execute_raw_sql(
            query_sql,
            params={"user_id": user_id, "user_status": "0"}
        )
        blocked_user_ids = [r["blocked_user_id"] for r in blocked_rows]
        
        # Build filters
        must_filters = [{"term": {"is_deleted": False}}]
        must_not_filters = []
        
        if blocked_user_ids:
            must_not_filters.append({"terms": {"user_id": blocked_user_ids}})
        
        # Search in Elasticsearch
        result_data = self.es_service.search_embedding(
            index_name,
            query_embedding,
            filters=must_filters,
            must_not_filters=must_not_filters,
            min_score=min_score,
            from_=0,
            size=limit,
            source_fields=["media_id", "name", "description", "ai_description", "tags", "user_id"]
        )
        
        # Get media_ids from ES results
        media_ids = []
        es_media_map = {}
        for hit in result_data.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            media_id = source.get("media_id")
            if media_id:
                media_ids.append(media_id)
                es_media_map[media_id] = {
                    "source": source,
                    "score": hit.get("_score", 0.0)
                }
        
        # Query DB to get media_url for all media_ids in one query (fix N+1)
        media_urls_map = {}
        if media_ids:
            # Build IN clause with placeholders
            placeholders = ",".join([f":id_{i}" for i in range(len(media_ids))])
            query_db = f"""
                SELECT id, media_url
                FROM medias
                WHERE id IN ({placeholders})
                and deleted_at is null
            """
            # Build params dict
            params = {f"id_{i}": media_id for i, media_id in enumerate(media_ids)}
            
            # Execute single query for all media_ids
            db_results = mysql.execute_raw_sql(
                query_db, 
                params=params,
                fetch_all=True
            )
            
            # Process results
            for db_result in db_results:
                media_id_str = str(db_result.get("id"))
                media_url = db_result.get("media_url")
                
                # Parse media_url (can be JSON string, list, or single string)
                if isinstance(media_url, str):
                    try:
                        media_url = json.loads(media_url)
                    except json.JSONDecodeError:
                        # If not JSON, treat as plain string
                        pass
                
                # Normalize to get first URL if it's a list
                if isinstance(media_url, list):
                    media_url = media_url[0] if media_url else ""
                elif not isinstance(media_url, str):
                    media_url = ""
                
                media_urls_map[media_id_str] = media_url
        
        # Format results - keep full data for RAG context, but prepare simplified response
        media_list = []
        for media_id in media_ids:
            if media_id not in es_media_map:
                continue
                
            es_data = es_media_map[media_id]
            source = es_data["source"]
            score = es_data["score"]
            
            # Get media_url from DB (first one if list)
            media_url = media_urls_map.get(media_id, "")
            
            # Store full data for RAG context (title, description, ai_description needed for LLM, but not returned in response)
            media_list.append({
                "id": media_id,
                "media_url": media_url,
                "title": source.get("name", ""),
                "description": source.get("description", ""),
                "ai_description": source.get("ai_description", ""),
                "popularity_score": round(score, 3),
                "user_id": source.get("user_id")
            })
        
        return media_list

    def format_rag_context(self, media_list: List[Dict[str, Any]]) -> str:
        """
        Format media list into RAG context string for LLM.
        Includes title, description, and ai_description for better context.
        """
        if not media_list:
            return "Không tìm thấy media nào phù hợp."
        
        context_parts = []
        for i, media in enumerate(media_list, 1):
            title = media.get("title", "")
            description = media.get("description", "")
            ai_description = media.get("ai_description", "")
            
            # Build context with title, description, and ai_description
            media_info_parts = []
            
            if title:
                media_info_parts.append(f"Title: {title}")
            
            if description:
                media_info_parts.append(f"Description: {description[:200]}")
            
            if ai_description:
                media_info_parts.append(f"AI Description: {ai_description[:200]}")
            
            if media_info_parts:
                context_parts.append(f"Media {i}:\n" + "\n".join(media_info_parts))
            else:
                context_parts.append(f"Media {i}")
        
        return "\n".join(context_parts)
    
    def format_media_response(self, media_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format media list for API response - only return id and media_url (first image if list).
        """
        result = []
        for media in media_list:
            media_url = media.get("media_url", "")
            # If media_url is a list, get first item
            if isinstance(media_url, list):
                media_url = media_url[0] if media_url else ""
            elif not isinstance(media_url, str):
                media_url = ""
            
            result.append({
                "id": media.get("id"),
                "media_url": media_url
            })
        return result

    async def filter_media_with_llm(
        self,
        media_list: List[Dict[str, Any]],
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to filter and select only relevant media from the list.
        Returns filtered media list with only the most relevant items.
        """
        if not media_list:
            return []
        
        # Format media list with numbers for LLM to reference
        media_context = self.format_rag_context(media_list)
        
        system_prompt = """
        You are a media filtering assistant. Analyze the user's query and select ONLY the media that are truly relevant.
        
        ## TASK
        Review all provided media and select ONLY those that match the user's query intent.
        Be strict - only select media that are clearly relevant. If a media is only partially relevant or unclear, do NOT select it.
        
        ## OUTPUT FORMAT
        Return ONLY a valid JSON array of media numbers (1-based index):
        [1, 3, 5]
        
        Example: If Media 1, Media 3, and Media 5 are relevant, return [1, 3, 5]
        If no media are relevant, return []
        """
        
        user_prompt = f"""
        User Query: {user_query}
        
        Media List:
        {media_context}
        
        Analyze the user's query and select ONLY the media numbers that are truly relevant.
        Return JSON array of selected media numbers only (e.g., [1, 3, 5]).
        """
        
        history = conversation_history or []
        prompt = self.gemini_service.buildPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history
        )
        
        # Debug log
        print(f"\n🔍 [FILTER_MEDIA] Prompt:", flush=True)
        print(json.dumps(prompt, indent=2, ensure_ascii=False), flush=True)
        
        response_text = await self.gemini_service.textToText(prompt)
        
        print(f"🔍 [FILTER_MEDIA] Response: {response_text[:200]}...", flush=True)
        
        # Parse JSON response to get selected media numbers
        selected_indices = []
        try:
            # Try to extract JSON array from response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                selected_indices = json.loads(json_str)
                # Convert to 0-based indices and filter
                selected_indices = [idx - 1 for idx in selected_indices if isinstance(idx, int) and 1 <= idx <= len(media_list)]
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            print(f"⚠️ [FILTER_MEDIA] Error parsing LLM response: {e}", flush=True)
            # Fallback: return all media if parsing fails
            return media_list
        
        # Filter media_list based on selected indices
        if not selected_indices:
            # If LLM selected nothing, return empty list
            return []
        
        filtered_media = [media_list[i] for i in selected_indices if 0 <= i < len(media_list)]
        
        print(f"🔍 [FILTER_MEDIA] Selected {len(filtered_media)} out of {len(media_list)} media", flush=True)
        
        return filtered_media

    async def handle_search_media(
        self,
        user_message: str,
        user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Handle SEARCH_MEDIA intent: search and answer questions about media.
        """
        # Extract number if user asks for specific count
        user_requested_limit = None
        match = re.search(r'(\d+)', user_message)
        if match:
            user_requested_limit = int(match.group(1))
        
        # System retrieves more from RAG for better selection, but user can only get max 10
        rag_limit = 30  # Retrieve more for LLM to filter from
        user_limit = min(user_requested_limit, 10) if user_requested_limit else 10  # Max 10 for user
        
        # Check if user requested too many
        user_requested_too_many = user_requested_limit and user_requested_limit > 10
        
        # Retrieve media via RAG with adaptive min_score
        # Start with lower min_score to get more results, then filter
        media_list = await self.retrieve_media_rag(user_message, user_id, limit=rag_limit, min_score=0.7)
        
        if not media_list:
            return {
                "intent": "SEARCH_MEDIA",
                "answer": "Xin lỗi, tôi không tìm thấy media nào phù hợp với yêu cầu của bạn.",
                "media": []
            }
        
        # Use LLM to filter and select only relevant media
        filtered_media_list = await self.filter_media_with_llm(media_list, user_message, conversation_history)
        
        # If LLM filtered out everything, use original list (fallback)
        if not filtered_media_list:
            filtered_media_list = media_list[:5]  # Fallback to top 5
        
        # Limit to user's requested amount (max 10)
        media_list = filtered_media_list[:user_limit]
        
        # Format RAG context for LLM (use all retrieved for better context)
        rag_context = self.format_rag_context(media_list)
        
        # Generate answer using LLM with RAG context
        system_prompt = """
        You are a helpful media management assistant. Answer user questions about media using ONLY the provided RAG context.
        
        ## RULES
        - Use ONLY information from the RAG context provided
        - Do NOT invent or hallucinate data
        - Answer in Vietnamese
        - Be concise and product-oriented
        - Do NOT repeat title, description, or tags in your answer
        - Just provide a natural, conversational answer about the media
        
        ## IMPORTANT: USER PREFERENCE DETECTION
        - If the user asks for "chỉ cần ảnh", "chỉ trả ảnh", "không trả lời dài", "chỉ media", "only image", or similar requests for ONLY media/images without explanation:
          → Return ONLY the word "MEDIA_ONLY" (no other text)
        - Otherwise, provide a normal conversational answer
        
        ## LIMIT NOTIFICATION
        - If user requested more than 10 media, mention that you can only provide up to 10 media at a time
        """
        
        limit_notice = ""
        if user_requested_too_many:
            limit_notice = f"\n\nNote: User requested {user_requested_limit} media, but system can only provide up to 10 media at a time."
        
        user_prompt = f"""
        RAG Context (Media Data):
        {rag_context}
        
        User Question: {user_message}{limit_notice}
        
        Answer the user's question based ONLY on the RAG context above. Be natural and conversational. Do NOT list titles, descriptions, or tags.
        
        If user wants only media/images without explanation, return "MEDIA_ONLY" only.
        If user requested more than 10 media, politely mention the 10 media limit.
        """
        
        history = conversation_history or []
        prompt = self.gemini_service.buildPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history
        )
        
        # Debug log
        print(f"\n🔍 [SEARCH_MEDIA] Prompt:", flush=True)
        print(json.dumps(prompt, indent=2, ensure_ascii=False), flush=True)
        
        answer = await self.gemini_service.textToText(prompt)
        
        print(f"🔍 [SEARCH_MEDIA] Response: {answer[:200]}...", flush=True)
        
        # Check if LLM decided to return only media
        if answer.strip().upper() == "MEDIA_ONLY":
            if user_requested_too_many:
                answer = f"Tôi chỉ có thể cung cấp tối đa 10 media. Đây là 10 media bạn cần:"
            else:
                answer = "Đây là các media bạn cần:"
        
        return {
            "intent": "SEARCH_MEDIA",
            "answer": answer,
            "media": self.format_media_response(media_list)
        }

    async def handle_suggest_media(
        self,
        user_message: str,
        user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Handle SUGGEST_MEDIA intent: suggest media and ask for album confirmation.
        """
        # Extract number if user asks for specific count
        user_requested_limit = 10
        match = re.search(r'(\d+)', user_message)
        if match:
            user_requested_limit = int(match.group(1))
        
        # System retrieves more from RAG for better selection, but user can only get max 10
        rag_limit = 30  # Retrieve more for LLM to filter from
        user_limit = min(user_requested_limit, 10) if user_requested_limit else 10  # Default 10, max 10 for user
        
        # Check if user requested too many
        user_requested_too_many = user_requested_limit and user_requested_limit > 10
        
        # Retrieve media via RAG with adaptive min_score
        # Start with lower min_score to get more results, then filter
        media_list = await self.retrieve_media_rag(user_message, user_id, limit=rag_limit, min_score=0.6)
        
        if not media_list:
            return {
                "intent": "SUGGEST_MEDIA",
                "answer": "Xin lỗi, tôi không tìm thấy media nào phù hợp để gợi ý.",
                "media": [],
                "ask_confirmation": None
            }
        
        # Use LLM to filter and select only relevant media
        filtered_media_list = await self.filter_media_with_llm(media_list, user_message, conversation_history)
        
        # If LLM filtered out everything, use original list (fallback)
        if not filtered_media_list:
            filtered_media_list = media_list[:5]  # Fallback to top 5
        
        # Limit to user's requested amount (max 10)
        media_list = filtered_media_list[:user_limit]
        
        # Format RAG context for LLM
        rag_context = self.format_rag_context(media_list)
        
        # Generate answer using LLM with RAG context
        system_prompt = """
        You are a helpful media management assistant. Suggest media to users using ONLY the provided RAG context.
        
        ## RULES
        - Use ONLY information from the RAG context provided
        - Do NOT invent or hallucinate data
        - Answer in Vietnamese
        - Be concise and friendly
        - Do NOT repeat title, description, or tags in your answer
        - Just provide a natural, conversational suggestion
        
        ## IMPORTANT: USER PREFERENCE DETECTION
        - If the user asks for "chỉ cần ảnh", "chỉ trả ảnh", "không trả lời dài", "chỉ media", "only image", or similar requests for ONLY media/images without explanation:
          → Return ONLY the word "MEDIA_ONLY" (no other text)
        - Otherwise, provide a normal conversational suggestion
        
        ## LIMIT NOTIFICATION
        - If user requested more than 10 media, mention that you can only provide up to 10 media at a time
        """
        
        limit_notice = ""
        if user_requested_too_many:
            limit_notice = f"\n\nNote: User requested {user_requested_limit} media, but system can only provide up to 10 media at a time."
        
        user_prompt = f"""
        RAG Context (Media Data):
        {rag_context}
        
        User Request: {user_message}{limit_notice}
        
        Suggest media to the user based ONLY on the RAG context above. Be natural and conversational. Do NOT list titles, descriptions, or tags.
        
        If user wants only media/images without explanation, return "MEDIA_ONLY" only.
        If user requested more than 10 media, politely mention the 10 media limit.
        """
        
        history = conversation_history or []
        prompt = self.gemini_service.buildPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history
        )
        
        # Debug log
        print(f"\n🔍 [SUGGEST_MEDIA] Prompt:", flush=True)
        print(json.dumps(prompt, indent=2, ensure_ascii=False), flush=True)
        
        answer = await self.gemini_service.textToText(prompt)
        
        print(f"🔍 [SUGGEST_MEDIA] Response: {answer[:200]}...", flush=True)
        
        # Check if LLM decided to return only media
        if answer.strip().upper() == "MEDIA_ONLY":
            if user_requested_too_many:
                answer = f"Tôi chỉ có thể cung cấp tối đa 10 media. Đây là 10 media gợi ý:"
            else:
                answer = "Đây là các media gợi ý:"
        
        return {
            "intent": "SUGGEST_MEDIA",
            "answer": answer,
            "media": self.format_media_response(media_list),
            "ask_confirmation": {
                "action": "CREATE_ALBUM",
                "message": "Bạn có muốn tạo album từ các media này không?"
            }
        }

    async def handle_confirm_create_album(
        self,
        user_message: str,
        suggested_media_ids: List[str],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Handle CONFIRM_CREATE_ALBUM intent: generate album name and description.
        """
        if not suggested_media_ids:
            return {
                "intent": "CONFIRM_CREATE_ALBUM",
                "error": "Không có media nào để tạo album."
            }
        
        # Get media details for context
        media_docs = self.es_service.mget(index_name, suggested_media_ids)
        media_list = []
        for media_id, doc in media_docs.items():
            if doc:
                media_list.append({
                    "title": doc.get("name", ""),
                    "description": doc.get("description") or doc.get("ai_description", ""),
                    "tags": doc.get("tags", [])
                })
        
        # Generate album name and description using LLM
        system_prompt = """
        You are a media management assistant. Generate album name and description based on the provided media list.
        
        ## TASK
        Analyze the media list and generate:
        1. A concise, engaging album name (max 50 characters)
        2. A brief description (2-3 sentences)
        
        ## OUTPUT FORMAT
        Return ONLY a valid JSON object:
        {
            "name": "album name",
            "description": "album description"
        }
        """
        
        media_context = "\n".join([
            f"- {m.get('title', 'N/A')}: {m.get('description', '')[:100]}"
            for m in media_list[:10]  # Limit to first 10 for context
        ])
        
        user_prompt = f"""
        Media List:
        {media_context}
        
        Generate album name and description for this collection of media. Return JSON only.
        """
        
        history = conversation_history or []
        prompt = self.gemini_service.buildPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history
        )
        
        # Debug log
        print(f"\n🔍 [CONFIRM_CREATE_ALBUM] Prompt:", flush=True)
        print(json.dumps(prompt, indent=2, ensure_ascii=False), flush=True)
        
        response_text = await self.gemini_service.textToText(prompt)
        
        print(f"🔍 [CONFIRM_CREATE_ALBUM] Response: {response_text[:200]}...", flush=True)
        
        # Parse JSON response
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                album_data = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON found", response_text, 0)
        except json.JSONDecodeError:
            # Fallback
            album_data = {
                "name": "Album mới",
                "description": f"Album chứa {len(suggested_media_ids)} media"
            }
        
        return {
            "intent": "CONFIRM_CREATE_ALBUM",
            "action": "CREATE_ALBUM",
            "album": {
                "name": album_data.get("name", "Album mới"),
                "description": album_data.get("description", ""),
                "media_ids": suggested_media_ids
            }
        }

    async def create_media_via_api(
        self,
        file_url: str,
        media_name: str,
        description: str,
        tags_name: List[str],
        user_id: str,
        token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Call API to create media in the system.
        Returns the created media response with id.
        """
        try:
            # Download file from URL
            from App.Services.MediaIngestService import MediaIngestService
            upload_file = await asyncio.to_thread(MediaIngestService.download_as_uploadfile, file_url)
            
            if not upload_file:
                print(f"❌ Failed to download file from URL: {file_url}", flush=True)
                return None
            
            # Get IP_SERVICE from environment
            ip_service = os.getenv("IP_SERVICE", "127.0.0.1")
            api_url = f"http://{ip_service}:80/api/medias"
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            
            # Read file content
            await upload_file.seek(0)
            file_content = await upload_file.read()
            file_like = io.BytesIO(file_content)
            
            # Add file as array (media must be an array)
            data.add_field(
                "media[]",
                file_like,
                filename=upload_file.filename,
                content_type=upload_file.content_type or "image/jpeg"
            )
            
            # Add other fields
            data.add_field("media_name", media_name)
            data.add_field("description", description or "")
            data.add_field("is_created", "false")
            
            # Add tags as array (tags_name must be an array)
            # Add each tag separately with tags_name[] format
            for tag in tags_name:
                data.add_field("tags_name[]", tag)
            
            # Prepare headers with Bearer token
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            
            # Make API call
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, data=data, headers=headers) as resp:
                    # Accept both 200 (OK) and 201 (Created) as success
                    if resp.status not in [200, 201]:
                        error_text = await resp.text()
                        print(f"❌ API create media failed: {resp.status} - {error_text}", flush=True)
                        return None
                    
                    response_data = await resp.json()
                    print(f"✅ Media created successfully: {response_data.get('media', {}).get('id')}", flush=True)
                    return response_data
                    
        except Exception as e:
            print(f"❌ Error creating media via API: {str(e)}", flush=True)
            return None

    async def handle_create_media_from_input(
        self,
        user_message: str,
        user_id: str,
        file_url: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle CREATE_MEDIA_FROM_INPUT intent: analyze input, generate metadata, and create media.
        """
        # Extract URL from message if not provided
        if not file_url:
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, user_message)
            if urls:
                file_url = urls[0]
        
        if not file_url:
            return {
                "intent": "CREATE_MEDIA_FROM_INPUT",
                "error": "Không tìm thấy URL hoặc file để tạo media."
            }
        
        # Use MediaIngestService logic to generate metadata
        from App.Services.MediaIngestService import MediaIngestService
        
        try:
            # Process media URL to get description
            combined_description = await MediaIngestService.process_media_urls([file_url])
            
            if not combined_description:
                return {
                    "intent": "CREATE_MEDIA_FROM_INPUT",
                    "error": "Không thể phân tích media từ URL này."
                }
            
            # Generate metadata using Gemini
            metadata = await MediaIngestService.generate_metadata_from_description(
                combined_description,
                self.gemini_service
            )
            
            title = metadata.get("title", "")
            description = metadata.get("description", "")
            tags = metadata.get("tags", [])
            
            # Create media via API
            create_response = await self.create_media_via_api(
                file_url=file_url,
                media_name=title,
                description=description,
                tags_name=tags,
                user_id=user_id,
                token=token
            )
            
            if not create_response:
                return {
                    "intent": "CREATE_MEDIA_FROM_INPUT",
                    "error": "Không thể tạo media trong hệ thống."
                }
            
            # Extract media id from response
            created_media = create_response.get("media", {})
            media_id = created_media.get("id")
            
            if not media_id:
                return {
                    "intent": "CREATE_MEDIA_FROM_INPUT",
                    "error": "Tạo media thành công nhưng không nhận được ID."
                }
            
            # Generate natural response using LLM
            system_prompt = """
            You are a helpful media management assistant. The user just created a DRAFT media (bản nháp).
            
            ## TASK
            Generate a natural, friendly response in Vietnamese to inform the user that their media draft has been created.
            IMPORTANT: This is only a DRAFT - the user needs to check it in the draft section before creating the final version.
            
            ## RULES
            - Answer in Vietnamese
            - Be natural and friendly
            - Keep it concise (2-3 sentences)
            - Don't be too formal or robotic
            - IMPORTANT: Clearly mention that this is a DRAFT (bản nháp)
            - Remind the user to check the draft section to review before creating the final version
            - Be helpful and guide the user to the next step
            """
            
            user_prompt = f"""
            User just created a DRAFT media (bản nháp) with:
            - Title: {title}
            - Description: {description[:100] if description else 'N/A'}
            - Tags: {', '.join(tags[:5]) if tags else 'N/A'}
            
            Generate a natural, friendly response to inform the user that:
            1. The media DRAFT has been created successfully
            2. This is only a draft (bản nháp)
            3. They should check it in the draft section to review before creating the final version
            
            Be conversational and helpful.
            """
            
            history = conversation_history or []
            prompt = self.gemini_service.buildPrompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                history=history
            )
            
            # Generate answer using LLM
            answer = await self.gemini_service.textToText(prompt)
            
            print(f"🔍 [CREATE_MEDIA] LLM Response: {answer[:200]}...", flush=True)
            
            return {
                "intent": "CREATE_MEDIA_FROM_INPUT",
                "answer": answer.strip(),
                "media": [{
                    "id": media_id,
                    "media_url": created_media.get("media_url", file_url)
                }]
            }
        except Exception as e:
            print(f"❌ Error in handle_create_media_from_input: {str(e)}", flush=True)
            return {
                "intent": "CREATE_MEDIA_FROM_INPUT",
                "error": f"Lỗi khi xử lý media: {str(e)}"
            }

    async def handle_general_qa(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Handle GENERAL_QA intent: answer general questions.
        """
        system_prompt = """
        You are a helpful media management assistant of system PinCap. Answer user questions about the media management system.
        
        ## RULES
        - Answer in Vietnamese
        - Be concise and helpful
        - If you don't know something, say so
        - Focus on media management features
        """
        
        user_prompt = user_message
        
        history = conversation_history or []
        prompt = self.gemini_service.buildPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history
        )
        
        # Debug log
        print(f"\n🔍 [GENERAL_QA] Prompt:", flush=True)
        print(json.dumps(prompt, indent=2, ensure_ascii=False), flush=True)
        
        answer = await self.gemini_service.textToText(prompt)
        
        print(f"🔍 [GENERAL_QA] Response: {answer[:200]}...", flush=True)
        
        return {
            "intent": "GENERAL_QA",
            "answer": answer
        }

    async def process_message(
        self,
        user_message: str,
        user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        suggested_media_ids: Optional[List[str]] = None,
        file_url: Optional[str] = None,
        token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point: process user message and return structured response.
        """
        # Detect intent
        intent = await self.detect_intent(user_message, conversation_history)
        
        # Route to appropriate handler
        if intent == "SEARCH_MEDIA":
            return await self.handle_search_media(user_message, user_id, conversation_history)
        
        elif intent == "SUGGEST_MEDIA":
            return await self.handle_suggest_media(user_message, user_id, conversation_history)
        
        elif intent == "CONFIRM_CREATE_ALBUM":
            if not suggested_media_ids:
                # Try to extract from conversation history or return error
                return {
                    "intent": "CONFIRM_CREATE_ALBUM",
                    "error": "Không có danh sách media để tạo album. Vui lòng yêu cầu gợi ý media trước."
                }
            return await self.handle_confirm_create_album(user_message, suggested_media_ids, conversation_history)
        
        elif intent == "CREATE_MEDIA_FROM_INPUT":
            return await self.handle_create_media_from_input(user_message, user_id, file_url, conversation_history, token)
        
        else:  # GENERAL_QA
            return await self.handle_general_qa(user_message, conversation_history)

