from confluent_kafka import Consumer, KafkaException
import os

class KafkaCore:
    """
    Core layer quản lý kết nối Kafka (dành cho cả consumer và producer sau này nếu cần).
    """

    def __init__(self, brokers: str):
        self.brokers = brokers

    # -----------------------------
    # 🔹 Tạo Kafka Consumer
    # -----------------------------
    def create_consumer(self, topic: str, group_id: str, offset_reset: str = "earliest") -> Consumer:
        """
        Trả về instance Kafka Consumer đã subscribe vào topic.
        Args:
            topic (str): Tên topic
            group_id (str): Kafka consumer group
            offset_reset (str): "earliest" | "latest"
        """
        try:
            conf = {
                "bootstrap.servers": self.brokers,
                "group.id": group_id,
                "auto.offset.reset": offset_reset,
                "max.poll.interval.ms": 60000
            }
            consumer = Consumer(conf)
            consumer.subscribe([topic])
            print(f"✅ Kafka consumer subscribed to topic: {topic}")
            return consumer
        except KafkaException as e:
            print(f"⚠️ Kafka Consumer init failed: {e}")
            raise


# Singleton core instance
url_kafka = os.getenv("IP_SERVICE")
port_kafka = os.getenv("KAFKA_PORT")
kafka_core = KafkaCore(brokers=f"{url_kafka}:{port_kafka}")
