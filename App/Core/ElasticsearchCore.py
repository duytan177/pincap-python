from elasticsearch import Elasticsearch
import os

class ElasticsearchCore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = cls._connect()
        return cls._instance

    @staticmethod
    def _connect():
        es_host = os.getenv("ES_HOST", f"http://{os.getenv('IP_SERVICE', 'localhost')}:9200")
        es_user = os.getenv("ES_USER")
        es_pass = os.getenv("ES_PASS")

        es = Elasticsearch(
            es_host,
            basic_auth=(es_user, es_pass) if es_user and es_pass else None,
            verify_certs=False,
        )

        if not es.ping():
            raise ConnectionError(f"❌ Cannot connect to Elasticsearch at {es_host}")
        print(f"✅ Connected to Elasticsearch at {es_host}")
        return es

# Singleton instance
es_client = ElasticsearchCore().client
