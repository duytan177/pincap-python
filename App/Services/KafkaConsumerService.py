import threading
import time
from App.Core.KafkaCore import kafka_core


class KafkaConsumerService:
    """
    Kafka consumer service linh hoạt:
    - Có thể xử lý từng event (real-time)
    - Hoặc gom nhiều event xử lý batch
    """

    def __init__(
        self,
        topic: str = "user_behavior",
        group_id: str = "behavior_analyzer_thread",
        handle_mode: str = "realtime",  # "realtime" hoặc "batch"
        batch_interval: int = 5,  # số giây gom 1 lượt
        batch_size: int = 20,  # số lượng message để trigger batch
        on_event=None,  # callback(event_str)
        on_batch=None,  # callback(list_of_events)
    ):
        self.topic = topic
        self.group_id = group_id
        self.handle_mode = handle_mode
        self.batch_interval = batch_interval
        self.batch_size = batch_size
        self.on_event = on_event
        self.on_batch = on_batch
        self._stop = False
        self._batch_buffer = []

    # -----------------------------
    # 🔹 Worker chính
    # -----------------------------
    def _consume_loop(self):
        print(f"🚀 Kafka consumer worker started in {self.handle_mode.upper()} mode ...")
        consumer = kafka_core.create_consumer(self.topic, self.group_id)

        last_batch_time = time.time()

        try:
            while not self._stop:
                msg = consumer.poll(1.0)
                if msg is None:
                    # Nếu ở batch mode, kiểm tra thời gian để flush batch
                    if (
                        self.handle_mode == "batch"
                        and self._batch_buffer
                        and time.time() - last_batch_time >= self.batch_interval
                    ):
                        self._flush_batch()
                        last_batch_time = time.time()
                    continue

                if msg.error():
                    print(f"⚠️ Kafka error: {msg.error()}")
                    continue

                event = msg.value().decode("utf-8")

                # 🔹 Realtime mode
                if self.handle_mode == "realtime" and callable(self.on_event):
                    self.on_event(event)

                # 🔹 Batch mode
                elif self.handle_mode == "batch":
                    self._batch_buffer.append(event)
                    if len(self._batch_buffer) >= self.batch_size:
                        self._flush_batch()
                        last_batch_time = time.time()

        except Exception as e:
            print(f"⚠️ Kafka consumer exception: {e}")
        finally:
            consumer.close()
            print("🛑 Kafka consumer stopped")

    # -----------------------------
    # 🔹 Flush batch
    # -----------------------------
    def _flush_batch(self):
        if self._batch_buffer and callable(self.on_batch):
            print(f"🧩 Flushing {len(self._batch_buffer)} messages as batch...")
            self.on_batch(self._batch_buffer)
        self._batch_buffer = []

    # -----------------------------
    # 🔹 Start in background
    # -----------------------------
    def start_background(self):
        t = threading.Thread(target=self._consume_loop, daemon=True)
        t.start()
        print("✅ Background Kafka consumer thread started.")

    # -----------------------------
    # 🔹 Stop consumer
    # -----------------------------
    def stop(self):
        self._stop = True
