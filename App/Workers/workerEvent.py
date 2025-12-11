from App.Services.KafkaConsumerService import KafkaConsumerService
import asyncio

def main():
    consumer_service = KafkaConsumerService(
        topic="user_events",
        group_id="worker_user_events",
        handle_mode="batch",
        batch_size=20,
        batch_timeout=60
    )

    asyncio.run(consumer_service.run())
if __name__ == "__main__":
   main()
