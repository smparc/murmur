import json
import torch
import torchaudio
from confluent_kafka import Consumer, Producer

# Configure CUDA Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kafka Configuration
KAFKA_BROKER = "localhost:9092"
RAW_TOPIC = "raw-audio-stream"
PROCESSED_TOPIC = "spectrogram-embeddings"

def create_kafka_clients():
    """Initialize highly-available Kafka consumer and producer."""
    consumer_conf = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': 'gpu-preprocessing-group',
        'auto.offset.reset': 'latest',
        'enable.auto.commit': True
    }
    producer_conf = {
        'bootstrap.servers': KAFKA_BROKER,
        'compression.type': 'lz4' # Essential for high-throughput tensor streaming
    }
    return Consumer(consumer_conf), Producer(producer_conf)

def get_mel_spectrogram_transform():
    """Initialize the STFT/Mel-Spectrogram transform directly on the GPU."""
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    ).to(device)

def process_stream():
    """Continuous ingestion, CUDA transformation, and downstream broadcasting."""
    consumer, producer = create_kafka_clients()
    consumer.subscribe([RAW_TOPIC])
    
    mel_transform = get_mel_spectrogram_transform()
    
    print(f"[*] Murmur CUDA Preprocessor listening on {RAW_TOPIC}...")

    try:
        while True:
            msg = consumer.poll(timeout=0.1)
            
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
            
            # 1. Extract metadata and raw audio bytes
            payload = json.loads(msg.value().decode('utf-8'))
            node_id = payload['node_id']
            timestamp = payload['timestamp']
            
            # Assuming payload['audio'] is a serialized float32 array
            raw_audio = torch.tensor(payload['audio'], dtype=torch.float32)
            
            # 2. Push to GPU & Compute Spectrogram
            raw_audio_gpu = raw_audio.to(device)
            spectrogram = mel_transform(raw_audio_gpu)
            
            # 3. Serialize and route to the ST-GNN topic
            # Note: In production, consider Protobuf or Apache Arrow for tensor serialization
            processed_payload = {
                "node_id": node_id,
                "timestamp": timestamp,
                "spectrogram": spectrogram.cpu().numpy().tolist()
            }
            
            producer.produce(
                PROCESSED_TOPIC, 
                key=node_id.encode('utf-8'), 
                value=json.dumps(processed_payload).encode('utf-8')
            )
            producer.poll(0) # Serve delivery callback queue

    except KeyboardInterrupt:
        print("[*] Terminating CUDA ingestion stream...")
    finally:
        consumer.close()
        producer.flush()

if __name__ == "__main__":
    # Ensure CUDA is available before spinning up the ingestion loop
    assert torch.cuda.is_available(), "CUDA is required for real-time preprocessing."
    process_stream()