import json
import re
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
            "max_output_tokens": 2048,
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
                and is_created = 1
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
            
            # Store full data for RAG context (description needed for LLM, but not returned in response)
            media_list.append({
                "id": media_id,
                "media_url": media_url,
                "description": source.get("description") or source.get("ai_description", ""),
                "popularity_score": round(score, 3),
                "user_id": source.get("user_id")
            })
        
        return media_list

    def format_rag_context(self, media_list: List[Dict[str, Any]]) -> str:
        """
        Format media list into RAG context string for LLM.
        Simplified format - only essential info.
        """
        if not media_list:
            return "Không tìm thấy media nào phù hợp."
        
        context_parts = []
        for i, media in enumerate(media_list, 1):
            description = media.get("description", "")
            # Only include description if it's meaningful (not empty)
            if description:
                context_parts.append(
                    f"Media {i}: {description[:100]}"
                )
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
        rag_limit = 50  # Retrieve more for better selection
        user_limit = min(user_requested_limit, 10) if user_requested_limit else 10  # Max 10 for user
        
        # Check if user requested too many
        user_requested_too_many = user_requested_limit and user_requested_limit > 10
        
        # Retrieve media via RAG with higher min_score to ensure quality
        # Use min_score=0.75 to only get highly relevant media
        media_list = await self.retrieve_media_rag(user_message, user_id, limit=rag_limit, min_score=0.75)
        
        if not media_list:
            return {
                "intent": "SEARCH_MEDIA",
                "answer": "Xin lỗi, tôi không tìm thấy media nào phù hợp với yêu cầu của bạn.",
                "media": []
            }
        
        # Only return up to user_limit (max 10), but if we found fewer, return what we have
        actual_count = len(media_list)
        media_list = media_list[:user_limit]
        final_count = len(media_list)
        
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
        - If you found fewer media than requested, naturally mention the actual count found
        """
        
        limit_notice = ""
        if user_requested_too_many:
            limit_notice = f"\n\nNote: User requested {user_requested_limit} media, but system can only provide up to 10 media at a time."
        elif user_requested_limit and final_count < user_requested_limit:
            limit_notice = f"\n\nNote: User requested {user_requested_limit} media, but only found {final_count} media with high relevance score."
        
        user_prompt = f"""
        RAG Context (Media Data):
        {rag_context}
        
        User Question: {user_message}{limit_notice}
        
        Answer the user's question based ONLY on the RAG context above. Be natural and conversational. Do NOT list titles, descriptions, or tags.
        
        If user wants only media/images without explanation, return "MEDIA_ONLY" only.
        If user requested more than 10 media, politely mention the 10 media limit.
        If fewer media were found than requested, naturally mention the actual count (e.g., "Tôi tìm thấy {final_count} media phù hợp").
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
                answer = f"Tôi chỉ có thể cung cấp tối đa 10 media. Đây là {final_count} media bạn cần:"
            elif user_requested_limit and final_count < user_requested_limit:
                answer = f"Tôi tìm thấy {final_count} media phù hợp với yêu cầu của bạn:"
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
        rag_limit = 50  # Retrieve more for better selection
        user_limit = min(user_requested_limit, 10) if user_requested_limit else 10  # Default 10, max 10 for user
        
        # Check if user requested too many
        user_requested_too_many = user_requested_limit and user_requested_limit > 10
        
        # Retrieve media via RAG with higher min_score to ensure quality
        # Use min_score=0.7 to get relevant media (slightly lower than search for suggestions)
        media_list = await self.retrieve_media_rag(user_message, user_id, limit=rag_limit, min_score=0.7)
        
        if not media_list:
            return {
                "intent": "SUGGEST_MEDIA",
                "answer": "Xin lỗi, tôi không tìm thấy media nào phù hợp để gợi ý.",
                "media": [],
                "ask_confirmation": None
            }
        
        # Only return up to user_limit (max 10), but if we found fewer, return what we have
        actual_count = len(media_list)
        media_list = media_list[:user_limit]
        final_count = len(media_list)
        
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
        - If you found fewer media than requested, naturally mention the actual count found
        """
        
        limit_notice = ""
        if user_requested_too_many:
            limit_notice = f"\n\nNote: User requested {user_requested_limit} media, but system can only provide up to 10 media at a time."
        elif user_requested_limit and final_count < user_requested_limit:
            limit_notice = f"\n\nNote: User requested {user_requested_limit} media, but only found {final_count} media with high relevance score."
        
        user_prompt = f"""
        RAG Context (Media Data):
        {rag_context}
        
        User Request: {user_message}{limit_notice}
        
        Suggest media to the user based ONLY on the RAG context above. Be natural and conversational. Do NOT list titles, descriptions, or tags.
        
        If user wants only media/images without explanation, return "MEDIA_ONLY" only.
        If user requested more than 10 media, politely mention the 10 media limit.
        If fewer media were found than requested, naturally mention the actual count (e.g., "Tôi tìm thấy {final_count} media phù hợp").
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
                answer = f"Tôi chỉ có thể cung cấp tối đa 10 media. Đây là {final_count} media gợi ý:"
            elif user_requested_limit and final_count < user_requested_limit:
                answer = f"Tôi tìm thấy {final_count} media phù hợp để gợi ý:"
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

    async def handle_create_media_from_input(
        self,
        user_message: str,
        file_url: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Handle CREATE_MEDIA_FROM_INPUT intent: analyze input and generate metadata.
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
            
            return {
                "intent": "CREATE_MEDIA_FROM_INPUT",
                "media": {
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "tags": metadata.get("tags", []),
                    "media_url": file_url
                },
                "frontend_link": "/media/create?prefill=true"
            }
        except Exception as e:
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
        Handle GENERAL_QA intent: answer general questions with rules.
        """
        system_prompt = """
        You are a helpful media management assistant of system PinCap. Answer user questions about the media management system.
        
        ## ABOUT PINCAP SYSTEM
        
        PinCap is a media management system that helps users organize, search, and manage their media content (images, videos, etc.). The system provides intelligent features powered by AI to make media management easier and more efficient.
        
        ## RULES FOR CONVERSATION
        
        ### 1. LANGUAGE & TONE
        - Always answer in Vietnamese
        - Use friendly, professional, and helpful tone
        - Be concise but informative
        - Use "bạn" (you) when addressing the user
        - Use "tôi" (I) when referring to yourself
        
        ### 2. CONTENT GUIDELINES
        - Focus on media management features and capabilities
        - Explain how to use the system features clearly
        - Provide step-by-step instructions when appropriate
        - If asked about something you don't know, politely say "Tôi không có thông tin về điều này" or "Tôi chưa được cập nhật về tính năng này"
        - Do NOT make up features or capabilities that don't exist
        - Do NOT provide information about other systems or unrelated topics
        
        ### 3. RESPONSE STRUCTURE
        - Start with a greeting if it's the first interaction
        - Answer the question directly
        - Provide examples when helpful
        - End with an offer to help further if appropriate
        
        ### 4. SYSTEM CAPABILITIES & FEATURES
        
        When users ask "Hệ thống này làm gì cho tôi?" or "PinCap làm gì?" or similar questions, explain:
        
        **Main Functions:**
        1. **Tìm kiếm Media thông minh**: 
           - Tìm kiếm media bằng từ khóa hoặc mô tả
           - Sử dụng AI để hiểu ý định của bạn
           - Ví dụ: "Tìm 10 media về robot", "Liệt kê media phổ biến nhất"
        
        2. **Gợi ý Media tự động**:
           - Hệ thống sẽ gợi ý media phù hợp với sở thích của bạn
           - Dựa trên chủ đề, tags, hoặc mô tả
           - Ví dụ: "Gợi ý 20 media về anime One Piece"
        
        3. **Tạo Album tự động**:
           - Tự động tạo album từ các media được gợi ý
           - Hệ thống tự động đặt tên và mô tả album
           - Giúp bạn tổ chức media một cách thông minh
        
        4. **Tạo Media từ URL**:
           - Thêm media mới vào hệ thống từ URL
           - Tự động phân tích và tạo metadata (tiêu đề, mô tả, tags)
           - Ví dụ: "Tạo media từ URL này: https://..."
        
        5. **Quản lý Media**:
           - Lưu trữ và tổ chức media của bạn
           - Tìm kiếm nhanh chóng với AI
           - Lọc media theo chủ đề, tags, hoặc mô tả
        
        **How to use:**
        - Simply chat with me naturally in Vietnamese
        - Ask me to search, suggest, or create media
        - I'll help you manage your media collection efficiently
        
        ### 5. EXAMPLE RESPONSES
        
        **If asked "Hệ thống này làm gì cho tôi?" or similar:**
        "Hệ thống PinCap giúp bạn quản lý media một cách thông minh. Tôi có thể giúp bạn:
        
        - **Tìm kiếm media**: Bạn có thể yêu cầu tôi tìm media theo chủ đề, ví dụ: 'Tìm 10 media về robot'
        
        - **Gợi ý media**: Tôi sẽ gợi ý các media phù hợp với sở thích của bạn, ví dụ: 'Gợi ý 20 media về anime'
        
        - **Tạo album**: Sau khi gợi ý media, tôi có thể giúp bạn tạo album tự động với tên và mô tả phù hợp
        
        - **Thêm media mới**: Bạn có thể thêm media từ URL, tôi sẽ tự động phân tích và tạo metadata
        
        Bạn muốn thử tính năng nào trước?"
        
        ### 6. BOUNDARIES
        - Do NOT answer questions about:
          * Personal information of other users
          * System technical details (database structure, API keys, etc.)
          * Unrelated topics (politics, religion, etc.)
        - If asked inappropriate questions, politely redirect: "Tôi chỉ có thể giúp bạn với các câu hỏi về quản lý media trong hệ thống PinCap"
        
        ### 7. CONVERSATION FLOW
        - Maintain context from conversation history
        - If user asks follow-up questions, use previous context
        - Be natural and conversational, not robotic
        - When explaining features, be enthusiastic but not overly promotional
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
        file_url: Optional[str] = None
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
            return await self.handle_create_media_from_input(user_message, file_url, conversation_history)
        
        else:  # GENERAL_QA
            return await self.handle_general_qa(user_message, conversation_history)

