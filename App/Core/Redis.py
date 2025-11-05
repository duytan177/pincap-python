import os
import asyncio
from redis.asyncio import Redis


class RedisCore:
    """
    Core kết nối Redis, chỉ dùng để tạo 1 instance duy nhất trong toàn app
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisCore, cls).__new__(cls)
            cls._instance._connect()
        return cls._instance

    def _connect(self):
        redis_host = os.getenv("REDIS_HOST", os.getenv("ENV_URL_SERVICE"))
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_password = os.getenv("REDIS_PASSWORD", "secret_redis")
        redis_db = int(os.getenv("REDIS_DB", 0))

        self.client = Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=redis_db,
            decode_responses=True,
        )
        print(f"✅ RedisCore connected to {redis_host}:{redis_port}/{redis_db}")

    async def ping(self):
        try:
            pong = await self.client.ping()
            return pong
        except Exception as e:
            print(f"❌ Redis ping failed: {e}")
            return False


# Singleton export
redis_core = RedisCore()
