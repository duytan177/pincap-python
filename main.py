import os
import json
import requests
import asyncio
from dotenv import load_dotenv
import time
import random
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv()

class GeminiService:
    def __init__(self, model: str, generationConfig: dict = None):
        """
        Initialize GeminiService with a model and optional generation config.
        """
        self.api_key = "AIzaSyAMMZKhZ7R-BTSp1vbHWz0k3vXXnojJP-o"
        if not self.api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in environment")

        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"

        # Default generation configuration
        self.generationConfig = generationConfig or {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024
        }

    async def textToText(self, messages: list) -> str:
        contents = [{"role": m.get("role", "user"), "parts": [{"text": m.get("content", "")}]} for m in messages]
        payload = {"contents": contents, "generationConfig": self.generationConfig}
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}?key={self.api_key}"

        for attempt in range(3):
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                data = response.json()
                parts = data["candidates"][0]["content"]["parts"]
                return "\n".join(p["text"] for p in parts if "text" in p).strip()
            elif response.status_code == 503:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Model overloaded. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")

        raise RuntimeError("Gemini API unavailable after retries.")


# Example usage
async def main():
    service = GeminiService("gemini-2.0-flash")  # ✅ text-only model
    messages = [
        {"role": "user", "content": "SYSTEM \n You are a helpful AI assistant. You are software engineer. Please answer questions clearly. Please response short and correctly center of gravity"},
        {"role": "user", "content": "Given me, code hello world of language programing c++"}
    ]
    print("🧩 Sending prompt to Gemini...")
    result = await service.textToText(messages)
    print("\n✨ AI Response:\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())


