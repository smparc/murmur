"""Integration test for the full pipeline (requires Kafka)."""


import pytest
import time


# Skip this entire module if Kafka is not available
pytest.importorskip("confluent_kafka")



@pytest.fixture
def kafka_available():
    """Check if Kafka broker is reachable."""
    from confluent_kafka import Producer
    try:
        p = Producer({"bootstrap.servers": "localhost:9092"})
        p.flush(timeout=2)
        return True
    except Exception:
        pytest.skip("Kafka broker not available")



class TestKafkaPipeline:
    @pytest.mark.integration
    def test_produce_and_consume_roundtrip(self, kafka_available):
        """Verify a message can roundtrip through Kafka."""
        import msgpack
        from confluent_kafka import Producer, Consumer


        topic = "murmur-test-roundtrip"
        test_payload = msgpack.packb({"test": True, "ts": time.time()}, use_bin_type=True)


        # Produce
        producer = Producer({"bootstrap.servers": "localhost:9092"})
        producer.produce(topic, value=test_payload)
        producer.flush(timeout=5)


        # Consume
        consumer = Consumer({
            "bootstrap.servers": "localhost:9092",
            "group.id": "test-group",
            "auto.offset.reset": "earliest",
        })
        consumer.subscribe([topic])


        msg = consumer.poll(timeout=10)
        consumer.close()


        assert msg is not None
        assert not msg.error()
        unpacked = msgpack.unpackb(msg.value(), raw=False)
        assert unpacked["test"] is True