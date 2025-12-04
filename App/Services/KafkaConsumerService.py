import json
import time
from App.Core.KafkaCore import kafka_core
from App.Services.MediaIngestService import MediaIngestService
import asyncio
import os
class KafkaConsumerService:
    """
    Kafka Consumer Service:
    - realtime mode: process each event immediately
    - batch mode: collect N events or after a timeout T seconds and then process
    """

    def __init__(
        self,
        topic: str,
        group_id: str,
        handle_mode: str = "realtime",  # "realtime" | "batch"
        batch_size: int = 10,
        batch_timeout: float = 5.0,  # seconds
        poll_timeout: float = 1.0,
    ):
        self.topic = topic
        self.group_id = group_id
        self.handle_mode = handle_mode
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.poll_timeout = poll_timeout

        self.consumer = kafka_core.create_consumer(topic, group_id)
        index_name = os.getenv("ELASTIC_SEARCH_INDEX")

        if not index_name:
            raise ValueError("ELASTIC_SEARCH_INDEX environment variable must be set")

        self.media_ingest_service = MediaIngestService(index_name)

    # -----------------------------------------
    # Handle a single event
    # -----------------------------------------
    async def handle_event(self, event: dict):
        print(f"📌 [Kafka Event] {event}", flush=True)
        await self.media_ingest_service.process_event(event=event)
        print("finished process event", flush=True)

    # -----------------------------------------
    # Handle batch of events
    # -----------------------------------------
    async def handle_batch(self, events: list):
        print(f"📌 [Kafka Batch] {len(events)} events: {events}", flush=True)
        # mediaIngest = MediaIngestService("media_embeddings_test_v3")
        # mediaIngest.process_batch(events=events)

    # -----------------------------------------
    # Main loop
    # -----------------------------------------
    async def run(self):
        print(f"🚀 Kafka worker started | topic={self.topic} | mode={self.handle_mode}", flush=True)
        try:
            if self.handle_mode == "realtime":
                await self._run_realtime()
            else:
                await self._run_batch()
        except Exception as e:
            print(f"❌ Kafka worker stopped due to error: {e}", flush=True)
        finally:
            print("⚠️ Closing Kafka consumer...", flush=True)
            self.consumer.close()

    # -----------------------------------------
    # Realtime mode: process each event immediately
    # -----------------------------------------
    async def _run_realtime(self):
        print("🚀 Starting realtime loop...", flush=True)
        while True:
            msg = self.consumer.poll(self.poll_timeout)
            if msg is None:
                continue
            if msg.error():
                print(f"⚠️ Kafka error: {msg.error()}", flush=True)
                continue
            try:
                data = json.loads(msg.value().decode("utf-8"))
                await self.handle_event(data)
                # Commit offset immediately
                self.consumer.commit(message=msg, asynchronous=False)
            except Exception as e:
                print(f"❌ Error processing event: {e}", flush=True)

    # -----------------------------------------
    # Batch mode: collect events and process together
    # -----------------------------------------
    async def _run_batch(self):
        buffer = []
        offset_buffer = []
        last_flush_time = time.time()

        print("🚀 Starting batch loop...", flush=True)
        while True:
            now = time.time()
            msg = self.consumer.poll(self.poll_timeout)

            if msg is None:
                # Check if timeout reached without new messages
                if buffer and now - last_flush_time >= self.batch_timeout:
                    await self._flush_batch(buffer, offset_buffer)
                    buffer = []
                    offset_buffer = []
                    last_flush_time = now
                await asyncio.sleep(0.5)
                continue

            if msg.error():
                print(f"⚠️ Kafka error: {msg.error()}", flush=True)
                continue

            try:
                data = json.loads(msg.value().decode("utf-8"))
                buffer.append(data)
                offset_buffer.append(msg)
            except Exception as e:
                print(f"❌ Error parsing event: {e}", flush=True)

            # Flush if batch full or timeout reached
            if buffer and (len(buffer) >= self.batch_size or now - last_flush_time >= self.batch_timeout):
                await self._flush_batch(buffer, offset_buffer)
                buffer = []
                offset_buffer = []
                last_flush_time = now

    # -----------------------------------------
    # Helper to flush batch and commit offsets
    # -----------------------------------------
    async def _flush_batch(self, buffer, offset_buffer):
        await self.handle_batch(buffer)
        for msg in offset_buffer:
            self.consumer.commit(message=msg, asynchronous=False)