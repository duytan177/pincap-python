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
        
        response = await self.gemini_service.textToText(prompt)
        intent = response.strip().upper()
        
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
        
        # Format results
        media_list = []
        for hit in result_data.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            score = hit.get("_score", 0.0)
            
            media_list.append({
                "id": source.get("media_id"),
                "title": source.get("name", ""),
                "description": source.get("description") or source.get("ai_description", ""),
                "tags": source.get("tags", []),
                "popularity_score": round(score, 3),
                "user_id": source.get("user_id")
            })
        
        return media_list

    def format_rag_context(self, media_list: List[Dict[str, Any]]) -> str:
        """
        Format media list into RAG context string for LLM.
        """
        if not media_list:
            return "Không tìm thấy media nào phù hợp."
        
        context_parts = []
        for i, media in enumerate(media_list, 1):
            title = media.get("title", "N/A")
            description = media.get("description", "")
            tags = media.get("tags", [])
            tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
            score = media.get("popularity_score", 0.0)
            
            context_parts.append(
                f"Media {i}:\n"
                f"- ID: {media.get('id')}\n"
                f"- Title: {title}\n"
                f"- Description: {description}\n"
                f"- Tags: {tags_str}\n"
                f"- Popularity Score: {score}\n"
            )
        
        return "\n".join(context_parts)

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
        limit = 10
        match = re.search(r'(\d+)', user_message)
        if match:
            limit = min(int(match.group(1)), 50)  # Max 50
        
        # Retrieve media via RAG
        media_list = await self.retrieve_media_rag(user_message, user_id, limit=limit)
        
        if not media_list:
            return {
                "intent": "SEARCH_MEDIA",
                "answer": "Xin lỗi, tôi không tìm thấy media nào phù hợp với yêu cầu của bạn.",
                "media": []
            }
        
        # Format RAG context
        rag_context = self.format_rag_context(media_list)
        
        # Generate answer using LLM with RAG context
        system_prompt = """
        You are a helpful media management assistant. Answer user questions about media using ONLY the provided RAG context.
        
        ## RULES
        - Use ONLY information from the RAG context provided
        - Do NOT invent or hallucinate data
        - Answer in Vietnamese
        - Be concise and product-oriented
        - List media with title and short description
        """
        
        user_prompt = f"""
        RAG Context (Media Data):
        {rag_context}
        
        User Question: {user_message}
        
        Answer the user's question based ONLY on the RAG context above. List the media with title and short description.
        """
        
        history = conversation_history or []
        prompt = self.gemini_service.buildPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history
        )
        
        answer = await self.gemini_service.textToText(prompt)
        
        return {
            "intent": "SEARCH_MEDIA",
            "answer": answer,
            "media": media_list
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
        limit = 20
        match = re.search(r'(\d+)', user_message)
        if match:
            limit = min(int(match.group(1)), 50)  # Max 50
        
        # Retrieve media via RAG
        media_list = await self.retrieve_media_rag(user_message, user_id, limit=limit, min_score=0.65)
        
        if not media_list:
            return {
                "intent": "SUGGEST_MEDIA",
                "answer": "Xin lỗi, tôi không tìm thấy media nào phù hợp để gợi ý.",
                "media": [],
                "ask_confirmation": None
            }
        
        # Format RAG context
        rag_context = self.format_rag_context(media_list)
        
        # Generate answer using LLM with RAG context
        system_prompt = """
        You are a helpful media management assistant. Suggest media to users using ONLY the provided RAG context.
        
        ## RULES
        - Use ONLY information from the RAG context provided
        - Do NOT invent or hallucinate data
        - Answer in Vietnamese
        - Be concise and friendly
        - Present the suggested media list
        """
        
        user_prompt = f"""
        RAG Context (Media Data):
        {rag_context}
        
        User Request: {user_message}
        
        Suggest media to the user based ONLY on the RAG context above. Present the suggestions in a friendly way.
        """
        
        history = conversation_history or []
        prompt = self.gemini_service.buildPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history
        )
        
        answer = await self.gemini_service.textToText(prompt)
        
        return {
            "intent": "SUGGEST_MEDIA",
            "answer": answer,
            "media": media_list,
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
        
        response_text = await self.gemini_service.textToText(prompt)
        
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
        Handle GENERAL_QA intent: answer general questions.
        """
        system_prompt = """
        You are a helpful media management assistant. Answer user questions about the media management system.
        
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
        
        answer = await self.gemini_service.textToText(prompt)
        
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

