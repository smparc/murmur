# Murmur
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)]()
[![Kubernetes](https://img.shields.io/badge/kubernetes-production-326ce5.svg)]()
[![Next.js](https://img.shields.io/badge/Next.js-Dashboard-black?logo=next.js)]()

**Murmur** is an enterprise-grade, spatio-temporal acoustic monitoring system. It turns ambient mechanical noise into a predictive maintenance engine. By ingesting continuous, multi-channel audio feeds from a sparse grid of microphones, Murmur localizes anomalous sounds, translates them into human-readable telemetry using an Audio LLM, and dynamically forecasts cascading equipment failures using Liquid Neural Networks (LNNs). 

Designed to be shipped to production environments rather than existing as a local proof of concept, the system leverages high-performance GPU compute, containerized orchestration, continuous CI/CD, and a real-time React dashboard to handle massive audio streams in real time.

---

## System Architecture

The following diagram illustrates the continuous data flow from physical audio capture to predictive text telemetry.

```mermaid
graph TD
    subgraph Edge / Factory Floor
        M1((Mic 1)) --> |Raw Audio| K[Apache Kafka Stream]
        M2((Mic 2)) --> |Raw Audio| K
        Sim[Mock Edge Simulator] -.-> K
    end

    subgraph GPU Accelerated Ingestion
        K --> C{CUDA / cuDF Preprocessing}
        C --> |Mel-Spectrograms| ST[Spatio-Temporal GNN]
    end

    subgraph Production Inference Cluster
        ST --> |Spatial/Temporal Embeddings| LLM[Audio LLM via vLLM/FastAPI]
        ST --> |Continuous Acoustic Data| LNN[Liquid Neural Network]
    end

    subgraph MLOps & Orchestration
        Dagster[Dagster Data Lineage] -.-> |Monitors| K
        Train[Training Pipeline] --> |Saves .pth| ST
        Train --> |Saves .pth| LNN
        Train -.-> |Logs Metrics| MLflow[MLflow Model Registry]
    end

    subgraph Output Routing
        LLM --> |Autoregressive Text Logs| UI([Next.js React Dashboard])
        LNN --> |Dynamic TTF Forecasts| UI
    end
```

---

## Technology Stack

| Component | Technology | Purpose in Production |
| :--- | :--- | :--- |
| **Data Ingestion** | Apache Kafka | Handles continuous, high-throughput raw audio streams without packet loss. |
| **Preprocessing** | Custom CUDA / RAPIDS | Bypasses CPU bottlenecks; extracts high-dimensional mel-spectrograms directly on the GPU. |
| **Feature Extraction** | ST-GNN (PyTorch Geo) | Models the physical facility as a topological graph to localize sound sources and track temporal frequency shifts. |
| **Telemetry Translation**| Multimodal Audio LLM | Acts as an autoregressive decoder, streaming text logs of physical anomalies (e.g., *"Impeller cavitation detected"*). |
| **Model Serving** | vLLM / FastAPI | Exposes the LLM, utilizing projection adapters and continuous batching to minimize Time-to-First-Token (TTFT). |
| **Failure Prediction** | Liquid Neural Networks | Adapts to drifting degradation patterns continuously via ODEs to forecast Time-to-Failure (TTF). |
| **Orchestration & Ops** | Dagster & MLflow | Tracks data lineage, pipeline health, and model drift over time. |
| **Deployment** | Docker & Kubernetes | Containerizes microservices and auto-scales inference pods dynamically based on acoustic energy spikes. |
| **Frontend** | React, Next.js, Recharts | Real-time visualization of system health, text logs, and predictive alerts. |

---

## Repository Structure

```text
murmur/
├── .github/
│   └── workflows/
│       └── deploy.yml                 # Automated CI/CD for Docker builds & K8s deployment
├── deploy/
│   ├── Dockerfile.ingest              # Container for CUDA audio preprocessing
│   ├── Dockerfile.inference           # Container for ST-GNN, LNN, and LLM serving
│   └── k8s/
│       ├── 01-kafka-cluster.yaml      # Kafka KRaft mode StatefulSet
│       ├── 02-ingest-deployment.yaml  # GPU-accelerated ingestion pods
│       ├── 03-inference-deployment.yaml # Load-balanced inference server
│       └── 04-autoscaling-hpa.yaml    # Horizontal Pod Autoscaling rules
├── frontend/
│   ├── package.json                   # Next.js dependencies
│   └── app/
│       └── page.tsx                   # Live React dashboard for TTF and telemetry
├── orchestration/
│   ├── __init__.py
│   └── data_pipeline.py               # Dagster assets and drift monitoring schedules
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── cuda_stream_processor.py   # Kafka consumer and GPU dispatcher
│   │   ├── mock_edge_device.py        # Simulates real-time factory anomalies
│   │   └── stft_kernels.cu            # Custom C++ CUDA kernels (Pre-emphasis & Hann)
│   ├── mapping/
│   │   ├── __init__.py
│   │   ├── st_gnn_model.py            # Spatio-Temporal Graph Neural Network
│   │   └── topology_graph.py          # Physical room geometry configuration
│   ├── translation/
│   │   ├── __init__.py
│   │   └── llm_decoder.py             # FastAPI LLM inference and embedding adapter
│   ├── forecasting/
│   │   ├── __init__.py
│   │   └── liquid_network.py          # Continuous-time Closed-form Network (CfC)
│   └── training/
│       └── train_pipeline.py          # Backprop, MSE Loss, and weight serialization
├── docker-compose.kafka.yml           # Local Kafka broker for development
├── requirements.txt                   # Core Python dependencies (CUDA 12.x target)
└── README.md                          # Project documentation
```

---

## Execution Pipeline

The project execution is divided into distinct phases to ensure scalability and fault tolerance.

| Phase | Description | Key Deliverables |
| :--- | :--- | :--- |
| **1. Ingestion** | Raw audio is captured and piped into Kafka topics. Custom CUDA kernels process the waveform into spectrograms on the fly. | Multi-channel streaming pipeline, CUDA preprocessing module. |
| **2. Mapping** | The facility's geometry is mapped into an ST-GNN. The model learns spatial dependencies (machine distances) and temporal acoustic patterns. | Trained ST-GNN, topological acoustic embeddings. |
| **3. Translation**| The ST-GNN embeddings trigger the Audio LLM inference engine. The LLM processes the embeddings to generate human-readable diagnostics. | vLLM serving endpoint, streaming text telemetry logs. |
| **4. Forecasting**| The Liquid Neural Network ingests the continuous streams. Its internal equations adapt in real time to shifting acoustic profiles. | Dynamic TTF (Time-to-Failure) probability metrics. |
| **5. Operations**| Dagster and MLflow monitor the data streams and track the drift of the LNN predictions over time. | Validated data lineage and retrain triggers. |
| **6. Deployment** | All microservices are containerized. Kubernetes handles horizontal pod autoscaling (HPA) during loud acoustic anomaly events. | Dockerfiles, K8s deployment manifests, CI/CD, active cluster. |

---

## ⚙️ Getting Started

### Prerequisites
*   NVIDIA GPU (CUDA 12.x compatible)
*   Windows Subsystem for Linux (WSL2) with Hardware Virtualization enabled (if on Windows)
*   Docker & Docker Compose
*   Kubernetes (Minikube/Kind for local, managed K8s for production)
*   Node.js v18+

### Installation & Local Simulation

**1. Clone the repository**
```bash
git clone [https://github.com/smparc/murmur.git](https://github.com/smparc/murmur.git)
cd murmur
```

**2. Spin up the Kafka Event Stream**
```bash
docker-compose -f docker-compose.kafka.yml up -d
```

**3. Train the Models (Initialize Weights)**
```bash
python3 src/training/train_pipeline.py
```

**4. Boot the Streaming Pipeline (Requires 3 Terminals)**
```bash
# Terminal 1: Start the CUDA Preprocessor
python3 src/ingestion/cuda_stream_processor.py

# Terminal 2: Start the LLM Telemetry Server
uvicorn src.translation.llm_decoder:app --host 0.0.0.0 --port 8000

# Terminal 3: Simulate the Edge Microphones
python3 src/ingestion/mock_edge_device.py
```

**5. Launch the Live Dashboard**
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:3000` to view the telemetry.

### Production Deployment

**1. Build the Preprocessing and Inference Containers**
```bash
docker build -t murmur-ingest:latest -f deploy/Dockerfile.ingest .
docker build -t murmur-inference:latest -f deploy/Dockerfile.inference .
```

**2. Deploy to Kubernetes**
```bash
kubectl apply -f deploy/k8s/
```

**3. Verify Pod Health**
Ensure all services (Kafka brokers, ST-GNN extractors, and LLM serving engines) are running:
```bash
kubectl get pods -o wide
kubectl get hpa murmur-inference-hpa
```
