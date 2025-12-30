import json
import base64
from App.Core.Redis import redis_core
from App.Helpers.GeminiEmbedding import getEmbedding


class RedisService:
    """
    Service thao tác với Redis: cache embedding, reset TTL, ...
    """

    def __init__(self):
        self.redis = redis_core.client
        self.default_ttl = 60 * 30  # 30 phút

    # -----------------------------
    # 🔹 Generate key
    # -----------------------------
    def make_key(self, user_id: str, file_content: bytes) -> str:
        hash_b64 = base64.urlsafe_b64encode(file_content).decode("utf-8")
        return f"user_{user_id}_{hash_b64}"

    # -----------------------------
    # 🔹 Save embedding
    # -----------------------------
    async def save(self, user_id: str, file_content: bytes, embedding: list):
        key = self.make_key(user_id, file_content)
        value = json.dumps({"embedding": embedding})
        await self.redis.setex(key, self.default_ttl, value)
        return key

    # -----------------------------
    # 🔹 Get embedding
    # -----------------------------
    async def get(self, user_id: str, file_content: bytes):
        key = self.make_key(user_id, file_content)
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None

    # -----------------------------
    # 🔹 Reset TTL (optional)
    # -----------------------------
    async def refresh_ttl(self, user_id: str, file_content: bytes):
        key = self.make_key(user_id, file_content)
        await self.redis.expire(key, self.default_ttl)

    # -----------------------------
    # 🔹 Clear cache by pattern
    # -----------------------------
    async def clear(self, pattern="user_*"):
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)
            return len(keys)
        return 0

    # -----------------------------
    # 🔹 Get or generate Gemini embedding
    # -----------------------------
    async def get_or_create_embedding(self, user_id: str, file=None, text: str = None):
        """
        Tự động lấy embedding từ cache nếu có,
        nếu chưa có thì đọc file, tạo embedding mới bằng embedding_fn, lưu lại, rồi trả về.

        Args:
            user_id (str): ID người dùng
            file: FastAPI UploadFile hoặc bất kỳ đối tượng có phương thức read()

        Returns:
            list: embedding vector
        """
        content = await file.read()
        cached = await self.get(user_id, content)
        if cached:
            print("✅ Found cached embedding")
            return cached["embedding"]

        try:
            file.file.seek(0)
            # Tạo embedding mới nếu không có trong cache
            embedding = None
            if file:
                # Nếu là file, tạo embedding từ file
                embedding = await getEmbedding(file=file)
            elif text:
                # Nếu là text, tạo embedding từ text
                embedding = await getEmbedding(text=text)

            await self.save(user_id, content, embedding)
            return embedding
        except Exception as e:
            raise ValueError(f"Embedding generation failed: {e}")
