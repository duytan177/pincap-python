# app/core/mysql_service.py
import os
from threading import Lock
from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

class MySQLService:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_engine()
        return cls._instance

    def _init_engine(self):
        MYSQL_USER = os.getenv("MYSQL_USER", "root")
        MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
        MYSQL_HOST = os.getenv("MYSQL_HOST", os.getenv("ENV_URL_SERVICE", "localhost"))
        MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
        MYSQL_DB = os.getenv("MYSQL_DB", "pincap")

        DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

        self.engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600,
            echo=False,
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)

    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MySQL error: {e}")
        finally:
            db.close()

    def execute_raw_sql(self, query: str, params: dict | None = None, fetch_all=True):
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            if fetch_all:
                return [dict(row._mapping) for row in result]
            else:
                row = result.fetchone()
                return dict(row._mapping) if row else None

