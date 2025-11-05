from elasticsearch import Elasticsearch
from fastapi import FastAPI
from App.Routes import TextToText, TextToImage, SearchByMedia
from confluent_kafka import Consumer, KafkaException

from App.Services.KafkaConsumerService import KafkaConsumerService
from App.Services.MediaIngestService import MediaIngestService

app = FastAPI(title="Gemini AI FastAPI Service")

# include routes
app.include_router(TextToText.router)
app.include_router(TextToImage.router)
app.include_router(SearchByMedia.router)

# Kafka configuration
KAFKA_BROKERS = '172.21.1.244:9092'
TOPIC = 'user_behavior'
GROUP_ID = 'behavior_analyzer_thread'

def handle_event(event: str):
    print("📩 Realtime received:", event)

def handle_batch(events: list):
    # Delegate async processing to MediaIngestService within this thread
    import asyncio
    service = MediaIngestService()
    asyncio.run(service.process_batch(events))


@app.on_event("startup")
def startup_event():
    # kafka_service = KafkaConsumerService(
    #     topic=TOPIC,
    #     handle_mode="realtime",
    #     on_event=handle_event
    # )

    kafka_service = KafkaConsumerService(
        topic="user_behavior",
        group_id="media_ingest_thread",
        handle_mode="batch",
        batch_interval=60,
        batch_size=20,
        on_batch=handle_batch
    )
    kafka_service.start_background()

@app.get("/")
def root():
    return {"message": "Gemini AI FastAPI Service is running"}
