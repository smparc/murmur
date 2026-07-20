import time
import json
import numpy as np
from confluent_kafka import Producer

# Configuration
KAFKA_BROKER = "localhost:9092"
RAW_TOPIC = "raw-audio-stream"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # Send half-second audio chunks
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION)

def delivery_report(err, msg):
    """Callback triggered on successful/failed message delivery."""
    if err is not None:
        print(f"[!] Message delivery failed: {err}")

def generate_mock_audio(node_id: int, anomaly: bool = False):
    """
    Generates a 1D numpy array representing a raw audio waveform.
    Injects a high-frequency sine wave if an anomaly is triggered.
    """
    t = np.linspace(0, CHUNK_DURATION, SAMPLES_PER_CHUNK, endpoint=False)
    
    # Base factory ambient noise (low frequency rumble + white noise)
    base_noise = 0.5 * np.sin(2 * np.pi * 60 * t) + np.random.normal(0, 0.1, SAMPLES_PER_CHUNK)
    
    if anomaly:
        # Simulate a high-frequency bearing squeal (2000 Hz)
        squeal = 0.8 * np.sin(2 * np.pi * 2000 * t)
        audio = base_noise + squeal
    else:
        audio = base_noise
        
    # Convert to float32 list for JSON serialization
    return audio.astype(np.float32).tolist()

def run_edge_simulation():
    """Streams continuous audio to Kafka from 4 simulated nodes."""
    producer = Producer({'bootstrap.servers': KAFKA_BROKER})
    print(f"[*] Mock Edge Device booting up. Streaming to {KAFKA_BROKER}...")
    
    nodes = [0, 1, 2, 3] # Representing our 4 factory microphones
    loop_count = 0
    
    try:
        while True:
            # Trigger a synthetic machine failure every 20 loops on Node 2
            trigger_anomaly = (loop_count % 20 == 0)
            
            for node in nodes:
                is_anomalous = trigger_anomaly and node == 2
                audio_data = generate_mock_audio(node, anomaly=is_anomalous)
                
                payload = {
                    "node_id": node,
                    "timestamp": time.time(),
                    "audio": audio_data,
                    "is_anomalous_flag": is_anomalous # Purely for our own debugging
                }
                
                producer.produce(
                    RAW_TOPIC,
                    key=str(node).encode('utf-8'),
                    value=json.dumps(payload).encode('utf-8'),
                    callback=delivery_report
                )
            
            producer.poll(0)
            loop_count += 1
            
            # Wait before generating the next chunk to simulate real-time streaming
            time.sleep(CHUNK_DURATION)
            print(f"[*] Streamed chunk {loop_count} from all nodes. (Anomaly: {trigger_anomaly})")
            
    except KeyboardInterrupt:
        print("\n[*] Stopping edge simulation.")
    finally:
        producer.flush()

if __name__ == "__main__":
    run_edge_simulation()