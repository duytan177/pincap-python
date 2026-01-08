import os
import requests
import base64
import json
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

from fastapi import UploadFile
from App.Helpers.APIKeyManager import get_gemini_key_manager

load_dotenv()

class GeminiService:
    def __init__(self, model: str, generationConfig: dict|None = None, use_key_rotation: bool = True):
        """
        Khởi tạo GeminiService.
        
        Args:
            model: URL của model Gemini
            generationConfig: Config cho generation
            use_key_rotation: Có sử dụng key rotation khi gặp 429 không (mặc định: True)
        """
        self.key_manager = get_gemini_key_manager() if use_key_rotation else None
        
        if self.key_manager:
            self.api_key = self.key_manager.get_current_key()
        else:
            # Fallback về cách cũ nếu không dùng rotation
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            self.api_key = api_key

        self.base_url = model
        self.model = model
        self.use_key_rotation = use_key_rotation

        # Default config
        self.generationConfig = generationConfig or {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 512,
        }

    # -----------------------------
    # KEEP ORIGINAL buildPrompt()
    # -----------------------------
    def buildPrompt(
            self,
            system_prompt: str,
            user_prompt: str,
            history: List[Dict[str, str]] = None,
            files: List[UploadFile] = None
    ):
        contents = []

        # 1️⃣ System prompt
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"SYSTEM PROMPT\n{system_prompt}"}]
            })

        # 2️⃣ Nếu có ảnh — thêm vào parts của người dùng
        user_parts = []

        if files:
            for file in files:
                # đọc file bytes rồi encode
                content = file.file.read()  # dùng sync vì buildPrompt có thể sync
                img_base64 = base64.b64encode(content).decode("utf-8")
                user_parts.append({
                    "inline_data": {
                        "mime_type": file.content_type or "image/png",
                        "data": img_base64
                    }
                })

        # 3️⃣ Cuối cùng thêm user prompt text
        user_parts.append({"text": f"USER PROMPT\n{user_prompt}"})

        contents.append({
            "role": "user",
            "parts": user_parts
        })

        # 5️⃣ Nếu có lịch sử hội thoại
        if history:
            for turn in history:
                contents.append({
                    "role": turn["role"],
                    "parts": [{"text": turn["content"]}]
                })

        return contents

    # -----------------------------
    # INTERNAL HELPERS
    # -----------------------------

    async  def _call_gemini_api(self, payload: dict, max_retries: int = 8) -> dict:
        """
        Generic request caller for Gemini API với tự động retry khi gặp 429.
        
        Args:
            payload: Payload để gửi đến API
            max_retries: Số lần retry tối đa (mặc định: 5)
        """
        last_error = None
        
        for attempt in range(max_retries):
            # Lấy API key hiện tại (có thể đã được rotate)
            if self.use_key_rotation and self.key_manager:
                current_key = self.key_manager.get_current_key()
            else:
                current_key = self.api_key
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": current_key
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}?key={current_key}",
                    headers=headers,
                    data=json.dumps(payload),
                )
                
                # Nếu thành công
                if response.status_code == 200:
                    return response.json()
                
                # Nếu gặp lỗi 429 (rate limit) hoặc 503 (overloaded) và có key rotation
                if response.status_code in [429, 503] and self.use_key_rotation and self.key_manager:
                    error_type = "rate limit (429)" if response.status_code == 429 else "overloaded (503)"
                    print(f"⚠️ API key bị {error_type} ở lần thử {attempt + 1}/{max_retries}", flush=True)
                    
                    # Đánh dấu key hiện tại bị failed
                    self.key_manager.mark_key_failed(current_key)
                    
                    # Kiểm tra xem còn key nào không
                    if self.key_manager.get_available_count() == 0:
                        # Nếu hết key, reset và thử lại
                        print(f"🔄 Tất cả keys đều bị {error_type}, reset và thử lại...", flush=True)
                        self.key_manager.reset_failed_keys()
                    
                    # Rotate sang key tiếp theo (nếu còn key)
                    if attempt < max_retries - 1:  # Chỉ rotate nếu còn lần retry
                        new_key = self.key_manager.rotate_to_next_key()
                        self.api_key = new_key
                        last_error = f"{error_type} - đã chuyển sang key khác"
                        continue
                    else:
                        last_error = f"{error_type} - đã thử hết {max_retries} lần"
                
                # Các lỗi khác
                last_error = f"Gemini API error {response.status_code}: {response.text}"
                if response.status_code not in [429, 503]:  # Không retry cho các lỗi khác 429 và 503
                    raise RuntimeError(last_error)
                    
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)}"
                if attempt < max_retries - 1:
                    continue
                raise RuntimeError(last_error)
        
        # Nếu đã hết số lần retry
        raise RuntimeError(f"Đã thử {max_retries} lần nhưng vẫn gặp lỗi. Lỗi cuối: {last_error}")

    # -----------------------------
    # 📦 CALL GEMINI EMBEDDING API
    # -----------------------------
    async def call_gemini_api_embedding(self, text: str, dimension: int = 1536, max_retries: int = 8) -> List[float]:
        """
        Gọi Gemini Embedding API để sinh vector từ text.
        Dùng base_url truyền sẵn ở constructor.
        Hỗ trợ tự động retry với key rotation khi gặp 429.
        """
        if not text or not text.strip():
            raise ValueError("❌ Input text for embedding cannot be empty.")

        last_error = None
        
        for attempt in range(max_retries):
            # Lấy API key hiện tại (có thể đã được rotate)
            if self.use_key_rotation and self.key_manager:
                current_key = self.key_manager.get_current_key()
            else:
                current_key = self.api_key

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": current_key
            }

            payload = {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text}]},
                "output_dimensionality": dimension
            }

            try:
                response = requests.post(self.base_url, headers=headers, json=payload)

                # Nếu thành công
                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("embedding", {}).get("values")
                    if not embedding:
                        raise ValueError(f"⚠️ No embedding found in response: {json.dumps(data, indent=2)}")
                    return embedding

                # Nếu gặp lỗi 429 (rate limit) hoặc 503 (overloaded) và có key rotation
                if response.status_code in [429, 503] and self.use_key_rotation and self.key_manager:
                    error_type = "rate limit (429)" if response.status_code == 429 else "overloaded (503)"
                    print(f"⚠️ API key bị {error_type} ở embedding API - lần thử {attempt + 1}/{max_retries}", flush=True)
                    
                    # Đánh dấu key hiện tại bị failed
                    self.key_manager.mark_key_failed(current_key)
                    
                    # Kiểm tra xem còn key nào không
                    if self.key_manager.get_available_count() == 0:
                        # Nếu hết key, reset và thử lại
                        print(f"🔄 Tất cả keys đều bị {error_type}, reset và thử lại...", flush=True)
                        self.key_manager.reset_failed_keys()
                    
                    # Rotate sang key tiếp theo (nếu còn lần retry)
                    if attempt < max_retries - 1:
                        new_key = self.key_manager.rotate_to_next_key()
                        self.api_key = new_key
                        last_error = f"{error_type} - đã chuyển sang key khác"
                        continue
                    else:
                        last_error = f"{error_type} - đã thử hết {max_retries} lần"

                # Các lỗi khác
                last_error = f"[GeminiEmbeddingError] HTTP {response.status_code}: {response.text}"
                if response.status_code not in [429, 503]:  # Không retry cho các lỗi khác 429 và 503
                    raise RuntimeError(last_error)
                    
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)}"
                if attempt < max_retries - 1:
                    continue
                raise RuntimeError(last_error)

        # Nếu đã hết số lần retry
        raise RuntimeError(f"Đã thử {max_retries} lần nhưng vẫn gặp lỗi. Lỗi cuối: {last_error}")

    def _format_response(self, data: dict, mode: str = "text") -> Any:
        """Extract response data (text or image)"""
        if mode == "text":
            texts = []
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part:
                        texts.append(part["text"])
            return "\n".join(texts).strip() or "[Empty response]"

        elif mode == "image":
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    # Case 1: Image data
                    if "inlineData" in part:
                        img_base64 = part["inlineData"]["data"]

                        img_bytes = base64.b64decode(img_base64)
                        with open("/app/output.png", "wb") as f:  # /app là WORKDIR
                            f.write(img_bytes)

                        return img_base64

                    # Case 2: Text description
                    elif "text" in part:
                        print("📝 Description:", part["text"])
        elif mode == "embedding":
            # 🧠 Nếu model sinh ảnh hoặc mô tả ảnh
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    # ✅ Nếu là ảnh base64 thì bỏ qua (vì ta chỉ cần mô tả text)
                    if "inlineData" in part:
                        continue

                    # ✅ Nếu là text (mô tả ảnh) thì dùng text này cho embedding
                    if "text" in part:
                        description = part["text"].strip()
                        print(f"📝 Extracted caption for embedding: {description}")
                        return description
        else:
            raise ValueError(f"Unsupported response mode: {mode}")

    # -----------------------------
    # PUBLIC METHODS
    # -----------------------------

    async def textToText(self, prompt: dict) -> str:
        """Generate text output"""
        generation =  self.generationConfig
        payload = {
            "contents": prompt,
            "generationConfig": generation,
        }

        data = await self._call_gemini_api(payload)
        return self._format_response(data, mode="text")

    async def textToImage(
        self,
        prompt: str
    ) -> List[str]:

        payload = {
            "contents": prompt,
            "generationConfig": {
                "responseModalities": ["IMAGE"]
            },
        }

        data = await self._call_gemini_api(payload)
        return self._format_response(data, mode="image")

    async def textWithImageToDescription(
        self,
        prompt: str
    ) -> str:

        payload = {
            "contents": prompt,
            "generationConfig": {
                "responseModalities": ["TEXT"]
            },
        }
        data = await self._call_gemini_api(payload)
        return self._format_response(data, mode="embedding")

    # -----------------------------
    # 🔹 Internal: text embedding
    # -----------------------------
    async def _getTextEmbedding(self, text: str) -> List[float]:
        """Gọi Gemini Embedding API để sinh embedding từ text"""
        embedding = await self.call_gemini_api_embedding(text)
        print(embedding)
        return embedding
