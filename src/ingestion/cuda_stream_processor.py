"""
CUDA-accelerated Kafka stream processor with sliding window buffer.


Consumes raw audio from Kafka, computes mel-spectrograms on GPU,
maintains a per-node temporal buffer of SEQ_LENGTH frames, and publishes
full windowed sequences downstream for the ST-GNN and LNN.
"""


import logging
import time
from collections import defaultdict, deque


import msgpack
import numpy as np
import torch
import torchaudio
from confluent_kafka import Consumer, Producer, KafkaError


from src.settings import settings


log = logging.getLogger(__name__)


# Resolve compute device once at import
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ---------------------------------------------------------------------------
# Sliding Window Buffer (per-node temporal context)
# ---------------------------------------------------------------------------


class SlidingWindowBuffer:
    """
    Maintains a per-node circular buffer of spectrograms.


    The LNN and ST-GNN need SEQ_LENGTH frames of temporal context to produce
    meaningful predictions. Without this buffer, they receive single frames
    and their temporal modeling capacity is wasted.
    """


    def __init__(self, window_size: int, num_nodes: int):
        self.window_size = window_size
        self.buffers: dict[int, deque] = {
            i: deque(maxlen=window_size) for i in range(num_nodes)
        }
        self.timestamps: dict[int, deque] = {
            i: deque(maxlen=window_size) for i in range(num_nodes)
        }


    def push(self, node_id: int, spectrogram: np.ndarray, timestamp: float):
        """Add a spectrogram frame to the node's buffer."""
        if node_id not in self.buffers:
            self.buffers[node_id] = deque(maxlen=self.window_size)
            self.timestamps[node_id] = deque(maxlen=self.window_size)
        self.buffers[node_id].append(spectrogram)
        self.timestamps[node_id].append(timestamp)


    def is_ready(self, node_id: int) -> bool:
        """Check if the buffer has enough frames for a full window."""
        return len(self.buffers.get(node_id, [])) >= self.window_size


    def get_window(self, node_id: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the full temporal window for a node.


        Returns:
            spectrograms: (window_size, n_mels, time_frames)
            timespans: (window_size,) — inter-frame time deltas for the LNN
        """
        frames = list(self.buffers[node_id])
        ts = list(self.timestamps[node_id])


        spectrograms = np.stack(frames, axis=0)


        # Compute inter-frame time deltas (for CfC irregular time handling)
        deltas = np.diff(ts, prepend=ts[0])
        deltas[0] = deltas[1] if len(deltas) > 1 else 0.5  # First delta unknown


        return spectrograms, deltas.astype(np.float32)


    def get_latest(self, node_id: int) -> np.ndarray | None:
        """Get the most recent spectrogram frame (for anomaly scoring)."""
        buf = self.buffers.get(node_id)
        if buf and len(buf) > 0:
            return buf[-1]
        return None



def create_kafka_clients() -> tuple[Consumer, Producer]:
    """Initialize Kafka consumer and producer with production-grade config."""
    consumer_conf = {
        "bootstrap.servers": settings.KAFKA_BROKER,
        "group.id": settings.KAFKA_GROUP_ID,
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
        "session.timeout.ms": 30_000,
        "max.poll.interval.ms": 300_000,
    }
    producer_conf = {
        "bootstrap.servers": settings.KAFKA_BROKER,
        "compression.type": settings.KAFKA_COMPRESSION,
        "linger.ms": 5,
        "batch.num.messages": 100,
    }
    return Consumer(consumer_conf), Producer(producer_conf)



def get_mel_spectrogram_transform() -> torchaudio.transforms.MelSpectrogram:
    """Initialize the Mel-Spectrogram transform on the target device."""
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=settings.SAMPLE_RATE,
        n_fft=settings.N_FFT,
        hop_length=settings.HOP_LENGTH,
        n_mels=settings.N_MELS,
    ).to(device)



def _delivery_callback(err, msg):
    if err is not None:
        log.error("Delivery failed for %s: %s", msg.key(), err)



def process_stream():
    """Continuous Kafka → GPU spectrogram → windowed publish loop."""
    consumer, producer = create_kafka_clients()
    consumer.subscribe([settings.RAW_TOPIC])


    mel_transform = get_mel_spectrogram_transform()
    buffer = SlidingWindowBuffer(
        window_size=settings.SEQ_LENGTH,
        num_nodes=settings.NUM_NODES,
    )
    consecutive_errors = 0
    max_errors = 50
    frames_processed = 0


    log.info(
        "CUDA Preprocessor listening on %s (device=%s, window=%d frames)",
        settings.RAW_TOPIC,
        device,
        settings.SEQ_LENGTH,
    )


    try:
        while True:
            msg = consumer.poll(timeout=0.1)


            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                consecutive_errors += 1
                log.warning(
                    "Consumer error (%d/%d): %s",
                    consecutive_errors,
                    max_errors,
                    msg.error(),
                )
                if consecutive_errors >= max_errors:
                    log.critical("Too many consecutive errors, shutting down")
                    break
                continue


            consecutive_errors = 0


            try:
                payload = msgpack.unpackb(msg.value(), raw=False)
                node_id = payload["node_id"]
                timestamp = payload["timestamp"]
                audio_bytes = payload["audio"]


                # Validate audio data
                if not audio_bytes or len(audio_bytes) < 4:
                    log.warning("Empty audio from node %d, skipping", node_id)
                    continue


                # Deserialize float32 audio from raw bytes
                raw_audio = torch.frombuffer(
                    bytearray(audio_bytes), dtype=torch.float32
                ).clone()


                # Check for NaN/Inf
                if torch.isnan(raw_audio).any() or torch.isinf(raw_audio).any():
                    log.warning("NaN/Inf in audio from node %d, skipping", node_id)
                    continue


                # Push to GPU and compute spectrogram
                raw_audio_gpu = raw_audio.to(device)
                spectrogram = mel_transform(raw_audio_gpu)
                spec_np = spectrogram.cpu().numpy()


                # Add to sliding window buffer
                buffer.push(node_id, spec_np, timestamp)
                frames_processed += 1


                # Always publish the latest single frame (for real-time anomaly scoring)
                single_payload = msgpack.packb(
                    {
                        "node_id": node_id,
                        "timestamp": timestamp,
                        "spectrogram_shape": list(spec_np.shape),
                        "spectrogram": spec_np.tobytes(),
                        "window_ready": buffer.is_ready(node_id),
                    },
                    use_bin_type=True,
                )
                producer.produce(
                    settings.PROCESSED_TOPIC,
                    key=str(node_id).encode("utf-8"),
                    value=single_payload,
                    callback=_delivery_callback,
                )


                # When buffer is full, also publish the complete temporal window
                if buffer.is_ready(node_id):
                    window_specs, timespans = buffer.get_window(node_id)
                    window_payload = msgpack.packb(
                        {
                            "node_id": node_id,
                            "timestamp": timestamp,
                            "window_shape": list(window_specs.shape),
                            "window": window_specs.tobytes(),
                            "timespans": timespans.tobytes(),
                        },
                        use_bin_type=True,
                    )
                    producer.produce(
                        settings.PROCESSED_TOPIC + "-windowed",
                        key=str(node_id).encode("utf-8"),
                        value=window_payload,
                        callback=_delivery_callback,
                    )


                producer.poll(0)


                if frames_processed % 100 == 0:
                    log.info("Processed %d frames", frames_processed)


            except Exception:
                log.exception("Failed to process message")


    except KeyboardInterrupt:
        log.info("Terminating CUDA ingestion stream...")
    finally:
        consumer.close()
        producer.flush()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not torch.cuda.is_available():
        log.warning("CUDA not available — falling back to CPU (not recommended for production)")
    process_stream()