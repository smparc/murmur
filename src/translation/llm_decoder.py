"""
LLM Telemetry Decoder — FastAPI inference service.


Receives ST-GNN acoustic embeddings along with anomaly detection context
(scores, severity, TTF predictions), projects them into the LLM's latent
space, and generates human-readable diagnostic text.


The WebSocket payload now includes structured anomaly data for the dashboard
instead of relying on regex extraction from generated text.


Endpoints:
    POST /generate_telemetry  — single-shot inference
    GET  /health              — liveness / readiness probe
    GET  /metrics             — Prometheus metrics
    WS   /ws/telemetry        — real-time WebSocket stream for the dashboard
"""


import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional


import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer


from src.observability.metrics import (
    ACTIVE_WS_CLIENTS,
    ANOMALY_COUNT,
    ANOMALY_SCORE,
    TTF_PREDICTION,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    track_inference,
    track_latency,
)
from src.settings import settings


log = logging.getLogger(__name__)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TelemetryRequest(BaseModel):
    node_id: int
    timestamp: float
    gnn_embedding: list[float]
    # Structured anomaly context (from anomaly detector + LNN)
    anomaly_score: float = Field(0.0, ge=0.0, le=1.0)
    anomaly_severity: Literal["normal", "warning", "critical"] = "normal"
    ttf_prediction: float = Field(0.0, ge=0.0, le=1.0)
    is_anomaly: bool = False


    ("node_id")
    
    def validate_node_id(cls, v):
        if v < 0:
            raise ValueError("node_id must be non-negative")
        return v


    ("gnn_embedding")
    
    def validate_embedding_dim(cls, v):
        if len(v) != settings.GNN_EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding of dim {settings.GNN_EMBEDDING_DIM}, got {len(v)}"
            )
        return v



class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    uptime_seconds: float



# ---------------------------------------------------------------------------
# Projection adapter
# ---------------------------------------------------------------------------


class EmbeddingProjector(nn.Module):
    """
    Two-layer MLP that maps the ST-GNN acoustic embedding (256-d)
    into the LLM's token embedding space (2048-d).
    """


    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)



# ---------------------------------------------------------------------------
# Application state (populated during lifespan)
# ---------------------------------------------------------------------------


class _AppState:
    tokenizer: Optional[AutoTokenizer] = None
    llm_model: Optional[AutoModelForCausalLM] = None
    projector: Optional[EmbeddingProjector] = None
    boot_time: float = 0.0
    ready: bool = False
    # Connected WebSocket clients (guarded by ws_lock)
    ws_clients: list[WebSocket] = []
    ws_lock: asyncio.Lock = asyncio.Lock()



state = _AppState()




async def lifespan(app: FastAPI):
    """Load models into VRAM on startup, clean up on shutdown."""
    state.boot_time = time.time()


    log.info("Loading tokenizer and LLM (%s) …", settings.LLM_MODEL_NAME)
    state.tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME)
    state.llm_model = AutoModelForCausalLM.from_pretrained(
        settings.LLM_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )


    log.info("Initializing Acoustic Projection Adapter …")
    state.projector = EmbeddingProjector(
        settings.GNN_EMBEDDING_DIM, settings.LLM_HIDDEN_DIM
    ).to(DEVICE)
    # In production, load pre-trained weights:
    # state.projector.load_state_dict(torch.load("projector_weights.pth"))
    state.projector.eval()


    state.ready = True
    log.info("Murmur inference server ready on %s", DEVICE)


    yield  # ← application runs here


    # Cleanup
    log.info("Shutting down inference server …")
    state.ready = False
    del state.llm_model, state.projector, state.tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



app = FastAPI(title="Murmur LLM Telemetry Decoder", lifespan=lifespan)



# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ready" if state.ready else "loading",
        device=DEVICE,
        model_loaded=state.ready,
        uptime_seconds=round(time.time() - state.boot_time, 1),
    )



# ---------------------------------------------------------------------------
# REST endpoint
# ---------------------------------------------------------------------------


@app.post("/generate_telemetry")
async def generate_telemetry(request: TelemetryRequest):
    """Receive an acoustic embedding with anomaly context and generate diagnostic text."""
    if not state.ready:
        raise HTTPException(status_code=503, detail="Models still loading")


    try:
        # Update Prometheus metrics
        ANOMALY_SCORE.labels(node_id=str(request.node_id)).set(request.anomaly_score)
        TTF_PREDICTION.labels(node_id=str(request.node_id)).set(request.ttf_prediction)
        if request.is_anomaly:
            ANOMALY_COUNT.labels(
                node_id=str(request.node_id),
                severity=request.anomaly_severity,
            ).inc()


        raw_embedding = (
            torch.tensor([request.gnn_embedding], dtype=torch.float32).to(DEVICE)
        )


        with track_inference("embedding_projector"):
            with torch.no_grad():
                acoustic_prompt_embeds = state.projector(raw_embedding).to(torch.float16)


        # Construct prompt with actual anomaly context
        severity_desc = {
            "normal": "operating within normal parameters",
            "warning": "showing early signs of degradation",
            "critical": "exhibiting critical acoustic anomalies requiring immediate attention",
        }
        system_prompt = (
            f"System Diagnostic for Node {request.node_id}:\n"
            f"Status: Sensor is {severity_desc.get(request.anomaly_severity, 'unknown')}.\n"
            f"Anomaly score: {request.anomaly_score:.3f}. "
            f"Failure probability: {request.ttf_prediction:.1%}.\n"
            f"Analysis: "
        )
        inputs = state.tokenizer(system_prompt, return_tensors="pt").to(DEVICE)
        text_embeds = state.llm_model.get_input_embeddings()(inputs.input_ids)


        combined_embeds = torch.cat(
            [text_embeds, acoustic_prompt_embeds.unsqueeze(1)], dim=1
        )


        with track_inference("llm_generation"):
            outputs = state.llm_model.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                do_sample=True,
                pad_token_id=state.tokenizer.eos_token_id,
            )


        generated_text = state.tokenizer.decode(outputs[0], skip_special_tokens=True)


        # Structured result with real model outputs (not regex-derived)
        result = {
            "node_id": request.node_id,
            "timestamp": request.timestamp,
            "telemetry": generated_text.strip(),
            "anomaly": {
                "score": round(request.anomaly_score, 4),
                "severity": request.anomaly_severity,
                "is_anomaly": request.is_anomaly,
            },
            "ttf_prediction": round(request.ttf_prediction, 4),
        }


        # Broadcast to all connected WebSocket clients
        await _broadcast_ws(result)


        return result


    except Exception:
        log.exception("Telemetry generation failed")
        raise HTTPException(status_code=500, detail="Telemetry processing failed")



# ---------------------------------------------------------------------------
# Prometheus metrics endpoint
# ---------------------------------------------------------------------------


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )



# ---------------------------------------------------------------------------
# WebSocket — real-time dashboard feed
# ---------------------------------------------------------------------------


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    """
    WebSocket endpoint for the Next.js dashboard.
    Clients connect here to receive live telemetry JSON frames.
    """
    await websocket.accept()
    async with state.ws_lock:
        state.ws_clients.append(websocket)
    ACTIVE_WS_CLIENTS.inc()
    log.info("WebSocket client connected (%d total)", len(state.ws_clients))


    try:
        while True:
            # Keep connection alive; ignore inbound messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        async with state.ws_lock:
            state.ws_clients.remove(websocket)
        ACTIVE_WS_CLIENTS.dec()
        log.info("WebSocket client disconnected (%d remaining)", len(state.ws_clients))



async def _broadcast_ws(payload: dict):
    """Send a telemetry frame to every connected WebSocket client."""
    async with state.ws_lock:
        dead = []
        for ws in state.ws_clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            state.ws_clients.remove(ws)
            ACTIVE_WS_CLIENTS.dec()



if __name__ == "__main__":
    import uvicorn


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(app, host=settings.INFERENCE_HOST, port=settings.INFERENCE_PORT)