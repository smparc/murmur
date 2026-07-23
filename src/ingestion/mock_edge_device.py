"""
Mock edge device that simulates real-time factory microphone streams.


Generates synthetic audio chunks with realistic degradation patterns:
    - Stochastic anomaly injection (not deterministic every 20 loops)
    - Progressive degradation (gradual onset, not binary)
    - No data leakage (is_anomalous_flag removed from payloads)
    - Multiple fault signatures (bearing, cavitation, imbalance)
"""


import logging
import time
from enum import Enum


import msgpack
import numpy as np
from confluent_kafka import Producer


from src.settings import settings


log = logging.getLogger(__name__)



class FaultType(Enum):
    NONE = "none"
    BEARING = "bearing"       # High-frequency squeal
    CAVITATION = "cavitation"  # Broadband noise burst
    IMBALANCE = "imbalance"    # Low-frequency modulation



def _delivery_report(err, msg):
    """Callback triggered on successful/failed message delivery."""
    if err is not None:
        log.error("Message delivery failed: %s", err)



def generate_mock_audio(
    node_id: int,
    fault: FaultType = FaultType.NONE,
    severity: float = 0.0,
) -> bytes:
    """
    Generate a raw audio waveform as float32 bytes.


    Args:
        node_id: Microphone identifier
        fault: Type of simulated machinery fault
        severity: Fault intensity [0.0 = none, 1.0 = severe]


    Returns raw bytes for efficient MessagePack transport.
    """
    t = np.linspace(
        0, settings.CHUNK_DURATION, settings.SAMPLES_PER_CHUNK, endpoint=False
    )


    # Base factory ambient noise (low-frequency rumble + white noise)
    base_noise = 0.5 * np.sin(2 * np.pi * 60 * t) + np.random.normal(
        0, 0.1, settings.SAMPLES_PER_CHUNK
    )


    if fault == FaultType.BEARING and severity > 0:
        # High-frequency bearing squeal (2-4 kHz with harmonics)
        freq = 2000 + node_id * 200
        squeal = severity * 0.8 * np.sin(2 * np.pi * freq * t)
        # Add harmonic
        squeal += severity * 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        base_noise += squeal


    elif fault == FaultType.CAVITATION and severity > 0:
        # Broadband noise burst (pump cavitation)
        burst = severity * np.random.normal(0, 0.5, settings.SAMPLES_PER_CHUNK)
        # Add transient impulses
        n_impulses = max(1, int(severity * 5))
        for _ in range(n_impulses):
            pos = np.random.randint(0, settings.SAMPLES_PER_CHUNK)
            width = min(50, settings.SAMPLES_PER_CHUNK - pos)
            base_noise[pos : pos + width] += severity * 1.5


    elif fault == FaultType.IMBALANCE and severity > 0:
        # Low-frequency amplitude modulation (rotating imbalance)
        mod_freq = 15 + node_id * 3
        modulation = 1.0 + severity * 0.6 * np.sin(2 * np.pi * mod_freq * t)
        base_noise *= modulation


    return base_noise.astype(np.float32).tobytes()



def run_edge_simulation():
    """Stream continuous audio to Kafka from simulated microphone nodes."""
    producer = Producer({"bootstrap.servers": settings.KAFKA_BROKER})
    num_nodes = settings.NUM_NODES


    log.info(
        "Mock Edge Device booting — %d nodes streaming to %s",
        num_nodes,
        settings.KAFKA_BROKER,
    )


    loop_count = 0


    # Stochastic degradation state per node
    node_degradation = {i: 0.0 for i in range(num_nodes)}
    node_fault_type = {i: FaultType.NONE for i in range(num_nodes)}
    anomaly_probability = 0.03  # 3% chance per frame of starting degradation


    try:
        while True:
            for node in range(num_nodes):
                # Stochastic anomaly injection (not deterministic)
                if node_degradation[node] == 0.0:
                    if np.random.random() < anomaly_probability:
                        # Start new degradation event
                        node_fault_type[node] = np.random.choice(
                            [FaultType.BEARING, FaultType.CAVITATION, FaultType.IMBALANCE]
                        )
                        node_degradation[node] = 0.1  # Start mild
                        log.debug("Node %d: %s degradation started", node, node_fault_type[node].value)
                else:
                    # Progressive worsening (realistic ramp-up)
                    node_degradation[node] = min(1.0, node_degradation[node] + np.random.uniform(0.01, 0.05))


                    # Chance of self-recovery (minor issue resolves)
                    if node_degradation[node] < 0.3 and np.random.random() < 0.1:
                        node_degradation[node] = 0.0
                        node_fault_type[node] = FaultType.NONE


                audio_bytes = generate_mock_audio(
                    node, fault=node_fault_type[node], severity=node_degradation[node]
                )


                # Payload has NO anomaly label — the system must detect it
                payload = msgpack.packb(
                    {
                        "node_id": node,
                        "timestamp": time.time(),
                        "audio": audio_bytes,
                    },
                    use_bin_type=True,
                )


                producer.produce(
                    settings.RAW_TOPIC,
                    key=str(node).encode("utf-8"),
                    value=payload,
                    callback=_delivery_report,
                )


            producer.poll(0)
            loop_count += 1
            time.sleep(settings.CHUNK_DURATION)


            if loop_count % 10 == 0:
                log.info("Streamed chunk %d from %d nodes", loop_count, num_nodes)


    except KeyboardInterrupt:
        log.info("Stopping edge simulation.")
    finally:
        producer.flush()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_edge_simulation()