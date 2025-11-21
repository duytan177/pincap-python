from typing import List, Dict, Any
from App.Core.ElasticsearchCore import es_client
from elasticsearch.helpers import bulk

class ElasticsearchService:
    def __init__(self, index_name: str, mapping: Dict[str, Any]):
        self.client = es_client
        self.create_index(index_name, mapping)

    # 🔹 Tạo index
    def create_index(self, index_name: str, mapping: Dict[str, Any]):
        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(index=index_name, body=mapping)
            print(f"✅ Created index: {index_name}")
        else:
            print(f"ℹ️ Index '{index_name}' already exists")

    # 🔹 Insert 1 doc
    def insert_document(self, index: str, id: str, document: Dict[str, Any]):
        try:
            clean_doc = {k: v for k, v in document.items() if v is not None}
            res = self.client.index(index=index, id=id, document=clean_doc)
            print(f"✅ Inserted document {id or res['_id']} into index '{index}'", flush=True)
            return res
        except Exception as e:
            print(f"❌ Failed to insert document: {e}", flush=True)
            raise e


    # 🔹 Insert theo batch (chunk)
    def insert_bulk_documents(self, index: str, documents: List[Dict[str, Any]], chunk_size: int = 500):
        """
        ✅ Insert nhiều documents vào Elasticsearch theo từng chunk.
        Không kiểm tra index tồn tại. Bỏ qua field None.
        """
        try:
            total = len(documents)
            print(f"📦 Start bulk insert {total} documents into '{index}' ...")

            for i in range(0, total, chunk_size):
                chunk = documents[i:i + chunk_size]
                actions = [
                    {
                        "_index": index,
                        "_id": doc.get("media_id"),
                        "_source": {k: v for k, v in doc.items() if v is not None}
                    }
                    for doc in chunk
                ]

                success, _ = bulk(self.client, actions)
                print(f"✅ Inserted chunk {i // chunk_size + 1}: {len(chunk)} docs")

            print(f"🎯 Bulk insert completed for index '{index}'")

        except Exception as e:
            print(f"❌ Bulk insert failed: {e}")
            raise e

    # 🔹 Tìm kiếm vector
    def search_embedding(self, index: str, query_vector: List[float],  filters: List[dict] | None = None, must_not_filters: List[dict] | None = None , min_score: float|None = 0.8, from_: int|None = None, size: int|None = 20, source_fields: list[str] | None = None):
        body = {
            "query": {
                "bool": {
                    "filter": filters,
                    "must_not": must_not_filters,
                    "must": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "(cosineSimilarity(params.query_vector, 'embedding') + 1.0) / 2",
                                "params": {"query_vector": query_vector},
                            },
                        }
                    }
                }
            },
        }

        # 🔹 Chỉ thêm vào body khi có giá trị
        if min_score is not None:
            body["min_score"] = min_score
        if from_ is not None:
            body["from"] = from_
        if size is not None:
            body["size"] = size
        if source_fields is not None:
            body["_source"] = source_fields

        res = self.client.search(index=index, body=body)

        return res