"""Integration test for the full pipeline (requires Kafka)."""

import time

import pytest

from src.settings import settings

# Skip this entire module if the client library is not installed at all.
pytest.importorskip("confluent_kafka")


@pytest.fixture
def broker() -> str:
    """
    Probe the broker for real, and skip if it is not there.

    Constructing a ``Producer`` does not connect — librdkafka dials lazily — and
    ``flush()`` on an empty queue returns 0 whether or not a broker exists. So
    the previous version of this fixture always reported "available", and the
    test then failed on a missing message rather than skipping. ``list_topics``
    performs an actual metadata round trip, which is the thing being asked.
    """
    from confluent_kafka import KafkaException, Producer

    address = settings.KAFKA_BROKER
    try:
        Producer({"bootstrap.servers": address}).list_topics(timeout=3)
    except KafkaException:
        pytest.skip(f"No Kafka broker at {address}")
    return address


class TestKafkaPipeline:
    @pytest.mark.integration
    def test_produce_and_consume_roundtrip(self, broker):
        """Verify a message can roundtrip through Kafka."""
        import msgpack
        from confluent_kafka import Consumer, Producer

        topic = "murmur-test-roundtrip"
        test_payload = msgpack.packb({"test": True, "ts": time.time()}, use_bin_type=True)

        # Produce
        producer = Producer({"bootstrap.servers": broker})
        producer.produce(topic, value=test_payload)
        producer.flush(timeout=5)

        # Consume
        consumer = Consumer(
            {
                "bootstrap.servers": broker,
                "group.id": "test-group",
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe([topic])

        msg = consumer.poll(timeout=10)
        consumer.close()

        assert msg is not None
        assert not msg.error()
        unpacked = msgpack.unpackb(msg.value(), raw=False)
        assert unpacked["test"] is True
