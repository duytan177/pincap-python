import os
import requests
import base64
import json
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

from fastapi import UploadFile

load_dotenv()

class GeminiService:
    def __init__(self, model: str, generationConfig: dict|None = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.api_key = api_key
        self.base_url = model
        self.model = model

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
                    "parts": [turn["content"]]
                })

        return contents

    # -----------------------------
    # INTERNAL HELPERS
    # -----------------------------

    async  def _call_gemini_api(self, payload: dict) -> dict:
        """Generic request caller for Gemini API"""
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{self.base_url}?key={self.api_key}",
            headers=headers,
            data=json.dumps(payload),
        )
        if response.status_code != 200:
            raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")

        return response.json()

    # -----------------------------
    # 📦 CALL GEMINI EMBEDDING API
    # -----------------------------
    async def call_gemini_api_embedding(self, text: str, dimension: int = 768) -> List[float]:
        """
        Gọi Gemini Embedding API để sinh vector từ text.
        Dùng base_url truyền sẵn ở constructor.
        """
        if not text or not text.strip():
            raise ValueError("❌ Input text for embedding cannot be empty.")

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        payload = {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": text}]},
            "output_dimensionality": dimension
        }

        response = requests.post(self.base_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"[GeminiEmbeddingError] HTTP {response.status_code}: {response.text}")

        data = response.json()
        embedding = data.get("embedding", {}).get("values")
        if not embedding:
            raise ValueError(f"⚠️ No embedding found in response: {json.dumps(data, indent=2)}")

        return embedding

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
        payload = {
            "contents": prompt,
            "generationConfig": self.generationConfig,
        }

        data = self._call_gemini_api(payload)
        return self._format_response(data, mode="text")

    async def textToImage(
        self,
        prompt: str
    ) -> List[str]:

        payload = {
            "contents": prompt,
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"]
            },
        }

        data = self._call_gemini_api(payload)
        return self._format_response(data, mode="image")

    async def textWithImageToDescription(
        self,
        prompt: str
    ) -> str:

        payload = {
            "contents": prompt,
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"]
            },
        }
        data = await self._call_gemini_api(payload)
        print(data)
        return self._format_response(data, mode="embedding")

    # -----------------------------
    # 🔹 Internal: text embedding
    # -----------------------------
    async def _getTextEmbedding(self, text: str) -> List[float]:
        """Gọi Gemini Embedding API để sinh embedding từ text"""
        embedding = await self.call_gemini_api_embedding(text)
        print(embedding)
        return embedding
