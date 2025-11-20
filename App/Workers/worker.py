from App.Services.KafkaConsumerService import KafkaConsumerService
def main():
    consumer = KafkaConsumerService(
        topic="user_behavior",
        group_id="worker_user_behavior",
        handle_mode="batch",
        batch_size=5,
        batch_timeout=30,
        poll_timeout=5
    )
    consumer.run()

if __name__ == "__main__":
    main()
