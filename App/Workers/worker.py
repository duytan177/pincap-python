from App.Services.KafkaConsumerService import KafkaConsumerService
import asyncio

def main():
    consumer = KafkaConsumerService(
        topic="user_behavior",
        group_id="worker_user_behavior",
        handle_mode="realtime",
        poll_timeout=5
    )
    asyncio.run(consumer.run())
if __name__ == "__main__":
   main()
