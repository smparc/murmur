"""
LLM Telemetry Decoder — FastAPI inference service.

Receives ST-GNN acoustic embeddings together with detector context (anomaly
score, severity, TTF), projects the embedding into the LLM's latent space and
generates a human-readable diagnostic.

The WebSocket payload carries *structured* fields alongside the prose. The
dashboard renders severity and TTF from those fields, never by pattern-matching
the generated text — model output is not a stable interface, and a monitoring
UI that changes colour based on whether the LLM said "critical" is a liability.

Endpoints
---------
``POST /generate_telemetry``  single-shot inference
``GET  /health``              liveness / readiness probe
``GET  /ready``               strict readiness (503 until models are resident)
``GET  /metrics``             Prometheus exposition
``WS   /ws/telemetry``        real-time dashboard feed
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager, suppress
from typing import Annotated, Any, Literal

import torch
import torch.nn as nn
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from src.observability.metrics import (
    ACTIVE_WS_CLIENTS,
    ANOMALY_COUNT,
    ANOMALY_SCORE,
    ANOMALY_Z_SCORE,
    CONTENT_TYPE,
    END_TO_END_LATENCY,
    MODEL_LOADED,
    TTF_PREDICTION,
    render,
    track_inference,
    track_latency,
)
from src.settings import settings

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# How many recent frames a freshly-connected dashboard receives, so an operator
# opening the page mid-shift sees context instead of a blank screen.
REPLAY_BUFFER_SIZE = 50

# Sending to a wedged client must not stall the broadcast path.
WS_SEND_TIMEOUT_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TelemetryRequest(BaseModel):
    """One scored frame, submitted by the inference worker."""

    node_id: int
    timestamp: float
    gnn_embedding: list[float]
    anomaly_score: float = Field(0.0, ge=0.0, le=1.0)
    anomaly_severity: Literal["normal", "warning", "critical"] = "normal"
    ttf_prediction: float = Field(0.0, ge=0.0, le=1.0)
    is_anomaly: bool = False
    z_score: float = 0.0

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, v: int) -> int:
        if v < 0:
            raise ValueError("node_id must be non-negative")
        return v

    @field_validator("gnn_embedding")
    @classmethod
    def validate_embedding_dim(cls, v: list[float]) -> list[float]:
        if len(v) != settings.GNN_EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding of dim {settings.GNN_EMBEDDING_DIM}, got {len(v)}"
            )
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp must be non-negative")
        return v


class AnomalyBlock(BaseModel):
    score: float
    severity: Literal["normal", "warning", "critical"]
    is_anomaly: bool
    z_score: float


class TelemetryResponse(BaseModel):
    node_id: int
    timestamp: float
    telemetry: str
    anomaly: AnomalyBlock
    ttf_prediction: float
    generated: bool = Field(description="True when an LLM produced the text; False when templated.")


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    llm_enabled: bool
    uptime_seconds: float
    connected_clients: int


# ---------------------------------------------------------------------------
# Projection adapter
# ---------------------------------------------------------------------------


class EmbeddingProjector(nn.Module):
    """Maps a ST-GNN acoustic embedding into the LLM's token embedding space."""

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
# Application state
# ---------------------------------------------------------------------------


class _AppState:
    def __init__(self) -> None:
        self.tokenizer: Any = None
        self.llm_model: Any = None
        self.projector: EmbeddingProjector | None = None
        self.projector_trained: bool = False
        self.boot_time: float = time.time()
        self.ready: bool = False
        self.ws_clients: list[WebSocket] = []
        self.replay: deque[dict] = deque(maxlen=REPLAY_BUFFER_SIZE)
        # Created lazily: an asyncio.Lock built at import time binds to whichever
        # loop first awaits it, which breaks as soon as a second event loop
        # exists (every TestClient instance creates one).
        self._ws_lock: asyncio.Lock | None = None
        self._rate_buckets: dict[str, deque[float]] = defaultdict(deque)

    @property
    def ws_lock(self) -> asyncio.Lock:
        if self._ws_lock is None:
            self._ws_lock = asyncio.Lock()
        return self._ws_lock


state = _AppState()


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Reject unauthenticated writes when ``MURMUR_API_KEY`` is configured."""
    if not settings.AUTH_ENABLED:
        return
    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def enforce_rate_limit(request: Request) -> None:
    """
    Fixed-window limiter keyed by API key, falling back to client address.

    Generation occupies a GPU for the duration of a request, so an unbounded
    endpoint is both a denial-of-service and a cost-amplification vector. This
    is per-process; a multi-replica deployment should additionally limit at the
    ingress, which is noted in the deployment manifests.
    """
    limit = settings.RATE_LIMIT_PER_MINUTE
    if limit <= 0:
        return

    key = request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    bucket = state._rate_buckets[key]
    while bucket and now - bucket[0] > 60.0:
        bucket.popleft()

    if len(bucket) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({limit}/min)",
            headers={"Retry-After": "60"},
        )
    bucket.append(now)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


def _load_projector(projector: EmbeddingProjector) -> bool:
    """Load trained adapter weights if the training pipeline produced them."""
    path = os.path.join(settings.MODEL_DIR, "projector_weights.pth")
    if not os.path.exists(path):
        log.warning(
            "No projector weights at %s — the acoustic embedding is a RANDOM "
            "projection and generated text is NOT conditioned on audio. Run "
            "`murmur-train` before relying on this output.",
            path,
        )
        return False
    projector.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    log.info("Loaded trained projector weights from %s", path)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.boot_time = time.time()

    if settings.AUTH_ENABLED:
        log.info("API key authentication is ENABLED")
    else:
        log.warning(
            "MURMUR_API_KEY is unset — /generate_telemetry is UNAUTHENTICATED. "
            "Acceptable only on a trusted network."
        )

    if settings.LLM_ENABLED:
        log.info("Loading tokenizer and LLM (%s) ...", settings.LLM_MODEL_NAME)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            state.tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME)
            state.llm_model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL_NAME,
                dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
            )
            if DEVICE != "cuda":
                state.llm_model.to(DEVICE)
            state.llm_model.eval()
            MODEL_LOADED.labels(model="llm").set(1)
            log.info("LLM resident on %s", DEVICE)
        except Exception:
            # Degrade to templated telemetry rather than failing readiness:
            # structured anomaly data is the safety-critical payload, and the
            # prose is a convenience layered on top of it.
            log.exception("LLM failed to load — falling back to templated telemetry")
            state.tokenizer = state.llm_model = None
            MODEL_LOADED.labels(model="llm").set(0)
    else:
        log.info("LLM_ENABLED=false — serving templated telemetry")
        MODEL_LOADED.labels(model="llm").set(0)

    projector = EmbeddingProjector(settings.GNN_EMBEDDING_DIM, settings.LLM_HIDDEN_DIM).to(DEVICE)
    state.projector_trained = _load_projector(projector)
    projector.eval()
    state.projector = projector
    MODEL_LOADED.labels(model="projector").set(1 if state.projector_trained else 0)

    state.ready = True
    log.info("Murmur inference server ready on %s", DEVICE)

    yield

    log.info("Shutting down inference server ...")
    state.ready = False
    async with state.ws_lock:
        clients = list(state.ws_clients)
        state.ws_clients.clear()
    for ws in clients:
        with suppress(Exception):
            await ws.close(code=status.WS_1001_GOING_AWAY)
    ACTIVE_WS_CLIENTS.set(0)

    # Drop buffered telemetry: after a restart the resident models may differ,
    # and replaying pre-restart frames to a reconnecting dashboard would
    # present them as current readings.
    state.replay.clear()
    state._rate_buckets.clear()

    state.llm_model = state.projector = state.tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Murmur LLM Telemetry Decoder",
    version="1.0.0",
    lifespan=lifespan,
)

# Without this the dashboard's /health poll is blocked by the browser: the page
# is served from :3000 and the API from :8000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGIN_LIST,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness. Always 200 while the process is serving."""
    return HealthResponse(
        status="ready" if state.ready else "loading",
        device=DEVICE,
        model_loaded=state.ready,
        llm_enabled=state.llm_model is not None,
        uptime_seconds=round(time.time() - state.boot_time, 1),
        connected_clients=len(state.ws_clients),
    )


@app.get("/ready")
async def ready() -> dict[str, bool]:
    """
    Readiness. 503 until models are resident.

    Split from ``/health`` so Kubernetes can restart a hung process (liveness)
    without also pulling a still-warming pod out of rotation permanently.
    """
    if not state.ready:
        raise HTTPException(status_code=503, detail="Models still loading")
    return {"ready": True}


# ---------------------------------------------------------------------------
# Telemetry generation
# ---------------------------------------------------------------------------

_SEVERITY_DESCRIPTION = {
    "normal": "operating within normal parameters",
    "warning": "showing early signs of degradation",
    "critical": "exhibiting critical acoustic anomalies requiring immediate attention",
}


def _build_prompt(request: TelemetryRequest) -> str:
    return (
        f"System Diagnostic for Node {request.node_id}:\n"
        f"Status: Sensor is "
        f"{_SEVERITY_DESCRIPTION.get(request.anomaly_severity, 'in an unknown state')}.\n"
        f"Anomaly score: {request.anomaly_score:.3f} "
        f"(robust z={request.z_score:.2f}). "
        f"Failure probability: {request.ttf_prediction:.1%}.\n"
        f"Analysis: "
    )


def _template_telemetry(request: TelemetryRequest) -> str:
    """Deterministic fallback when no LLM is resident."""
    headline = {
        "normal": "Nominal acoustic signature.",
        "warning": "Deviation from baseline acoustic signature detected.",
        "critical": "Severe acoustic anomaly — inspection recommended.",
    }[request.anomaly_severity]
    return (
        f"Node {request.node_id}: {headline} "
        f"Anomaly score {request.anomaly_score:.3f} (z={request.z_score:.2f}); "
        f"modelled failure probability {request.ttf_prediction:.1%}."
    )


@torch.no_grad()
def _generate_text(request: TelemetryRequest) -> tuple[str, bool]:
    """Returns ``(text, generated_by_llm)``."""
    embedding = torch.tensor([request.gnn_embedding], dtype=torch.float32, device=DEVICE)

    with track_inference("embedding_projector"):
        acoustic_embeds = state.projector(embedding)

    if state.llm_model is None or state.tokenizer is None:
        return _template_telemetry(request), False

    target_dtype = state.llm_model.get_input_embeddings().weight.dtype
    acoustic_embeds = acoustic_embeds.to(target_dtype)

    prompt = _build_prompt(request)
    inputs = state.tokenizer(prompt, return_tensors="pt").to(state.llm_model.device)
    text_embeds = state.llm_model.get_input_embeddings()(inputs.input_ids)

    combined = torch.cat([text_embeds, acoustic_embeds.unsqueeze(1).to(text_embeds.device)], dim=1)
    # Every position is real; an explicit mask silences the transformers warning
    # and keeps behaviour correct if batching is added later.
    attention_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=combined.device)

    with track_inference("llm_generation"):
        outputs = state.llm_model.generate(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            do_sample=settings.LLM_TEMPERATURE > 0,
            pad_token_id=state.tokenizer.eos_token_id or state.tokenizer.pad_token_id,
        )

    # With inputs_embeds, generate() returns only the newly generated tokens.
    return state.tokenizer.decode(outputs[0], skip_special_tokens=True).strip(), True


@app.post(
    "/generate_telemetry",
    response_model=TelemetryResponse,
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
@track_latency("generate_telemetry")
async def generate_telemetry(request: TelemetryRequest) -> TelemetryResponse:
    """Score-annotated embedding in, diagnostic telemetry out."""
    if not state.ready:
        raise HTTPException(status_code=503, detail="Models still loading")

    node_label = str(request.node_id)
    ANOMALY_SCORE.labels(node_id=node_label).set(request.anomaly_score)
    ANOMALY_Z_SCORE.labels(node_id=node_label).set(request.z_score)
    TTF_PREDICTION.labels(node_id=node_label).set(request.ttf_prediction)
    if request.is_anomaly:
        ANOMALY_COUNT.labels(node_id=node_label, severity=request.anomaly_severity).inc()

    try:
        # Generation is synchronous and GPU-bound; off-loading keeps the event
        # loop free to service WebSocket traffic and health probes.
        text, generated = await asyncio.to_thread(_generate_text, request)
    except Exception:
        log.exception("Telemetry generation failed for node %s", request.node_id)
        raise HTTPException(status_code=500, detail="Telemetry processing failed") from None

    if request.timestamp > 0:
        age = time.time() - request.timestamp
        if 0 <= age < 3600:
            END_TO_END_LATENCY.observe(age)

    response = TelemetryResponse(
        node_id=request.node_id,
        timestamp=request.timestamp,
        telemetry=text,
        anomaly=AnomalyBlock(
            score=round(request.anomaly_score, 4),
            severity=request.anomaly_severity,
            is_anomaly=request.is_anomaly,
            z_score=round(request.z_score, 4),
        ),
        ttf_prediction=round(request.ttf_prediction, 4),
        generated=generated,
    )

    payload = response.model_dump()
    state.replay.append(payload)
    await _broadcast_ws(payload)
    return response


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(render().decode("utf-8"), media_type=CONTENT_TYPE)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket) -> None:
    """Live telemetry feed for the dashboard."""
    if settings.AUTH_ENABLED:
        supplied = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
        if supplied != settings.API_KEY:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    await websocket.accept()
    async with state.ws_lock:
        state.ws_clients.append(websocket)
    ACTIVE_WS_CLIENTS.inc()
    log.info("WebSocket client connected (%d total)", len(state.ws_clients))

    try:
        for frame in list(state.replay):
            await websocket.send_json(frame)

        while True:
            # The dashboard is receive-only; this simply parks until it goes.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        log.debug("WebSocket closed abnormally", exc_info=True)
    finally:
        # try/finally, because previously any exception other than a clean
        # WebSocketDisconnect leaked the client and left the gauge permanently
        # over-counted.
        async with state.ws_lock:
            if websocket in state.ws_clients:
                state.ws_clients.remove(websocket)
                ACTIVE_WS_CLIENTS.dec()
        log.info("WebSocket client disconnected (%d remaining)", len(state.ws_clients))


async def _broadcast_ws(payload: dict) -> None:
    """
    Fan a telemetry frame out to every connected client.

    The client list is snapshotted under the lock and the sends happen outside
    it, concurrently and with a timeout. Holding the lock across serial awaits —
    as this previously did — means one wedged client stalls every other client's
    updates *and* blocks new connections from being accepted.
    """
    async with state.ws_lock:
        clients = list(state.ws_clients)

    if not clients:
        return

    async def send(ws: WebSocket) -> WebSocket | None:
        try:
            await asyncio.wait_for(ws.send_json(payload), timeout=WS_SEND_TIMEOUT_SECONDS)
            return None
        except Exception:
            return ws

    results = await asyncio.gather(*(send(ws) for ws in clients), return_exceptions=True)
    dead = [r for r in results if isinstance(r, WebSocket)]

    if dead:
        async with state.ws_lock:
            for ws in dead:
                if ws in state.ws_clients:
                    state.ws_clients.remove(ws)
                    ACTIVE_WS_CLIENTS.dec()
        log.info("Dropped %d unresponsive WebSocket client(s)", len(dead))


def main() -> None:  # pragma: no cover
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(app, host=settings.INFERENCE_HOST, port=settings.INFERENCE_PORT)


if __name__ == "__main__":  # pragma: no cover
    main()
