"""
Tests for the streaming inference worker.

This is the service that joins ingestion to serving. It did not exist before —
the spectrogram topics were produced and never consumed — so none of this path
had any coverage at all.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from src.inference.worker import InferenceWorker, NodeWindow, WindowAssembler, decode_window
from src.settings import settings


def _window(node_id: int, timestamp: float, amplitude: float = 1.0, seq: int = 8) -> NodeWindow:
    rng = np.random.default_rng(node_id * 31 + int(timestamp))
    frames = (rng.standard_normal((seq, settings.N_MELS, 6)) * amplitude).astype(np.float32)
    return NodeWindow(
        node_id=node_id,
        features=frames.mean(axis=2),
        timespans=np.full(seq, 0.5, dtype=np.float32),
        timestamp=timestamp,
        latest_frame=frames[-1],
    )


class TestWindowAssembler:
    def test_incomplete_until_every_node_reports(self):
        assembler = WindowAssembler(num_nodes=4, seq_length=8, n_mels=settings.N_MELS)
        for node in range(3):
            assembler.push(_window(node, 100.0))
            assert not assembler.is_complete()
        assembler.push(_window(3, 100.0))
        assert assembler.is_complete()

    def test_stale_node_blocks_completion(self):
        """
        A microphone that dropped out an hour ago must not be stitched into a
        snapshot alongside live ones — the GCN would convolve across time.
        """
        assembler = WindowAssembler(
            num_nodes=2, seq_length=8, n_mels=settings.N_MELS, staleness_tolerance=5.0
        )
        assembler.push(_window(0, 100.0))
        assembler.push(_window(1, 100.0 + 60.0))
        assert not assembler.is_complete()

    def test_assembled_shape_matches_gnn_contract(self):
        assembler = WindowAssembler(num_nodes=4, seq_length=8, n_mels=settings.N_MELS)
        for node in range(4):
            assembler.push(_window(node, 100.0))

        x, timespans, snapshot = assembler.assemble()
        assert x.shape == (1, 8, 4 * settings.N_MELS)
        assert timespans.shape == (1, 8)
        assert set(snapshot) == {0, 1, 2, 3}

    def test_node_ordering_is_stable(self):
        """Node k must always occupy the same feature slice, or edges misalign."""
        assembler = WindowAssembler(num_nodes=2, seq_length=4, n_mels=settings.N_MELS)
        w0, w1 = _window(0, 100.0, seq=4), _window(1, 100.0, seq=4)

        assembler.push(w1)
        assembler.push(w0)
        x_a, _, _ = assembler.assemble()
        assembler.clear()

        assembler.push(w0)
        assembler.push(w1)
        x_b, _, _ = assembler.assemble()

        assert np.allclose(x_a.numpy(), x_b.numpy())

    def test_clear_resets(self):
        assembler = WindowAssembler(num_nodes=2, seq_length=4, n_mels=settings.N_MELS)
        assembler.push(_window(0, 100.0, seq=4))
        assembler.clear()
        assert not assembler.is_complete()

    def test_a_mismatched_features_length_is_rejected_at_the_door(self):
        assembler = WindowAssembler(num_nodes=2, seq_length=4, n_mels=settings.N_MELS)
        assert not assembler.push(_window(0, 100.0, seq=8))
        assert not assembler.is_complete()

    def test_a_mismatched_timespans_length_is_rejected_too(self):
        """Both arrays are indexed by seq_length, so both must be checked.

        `assemble` averages timespans across the reporting nodes and numpy
        raises on a ragged stack. Letting a bad-length timespans array through
        `push` moves the failure from one clear diagnostic here to a traceback
        from inside `assemble` on every subsequent snapshot, forever.
        """
        assembler = WindowAssembler(num_nodes=2, seq_length=4, n_mels=settings.N_MELS)
        window = _window(0, 100.0, seq=4)
        window.timespans = np.full(7, 0.5, dtype=np.float32)

        assert not assembler.push(window)
        assert not assembler.is_complete()

    def test_assemble_never_sees_ragged_timespans(self):
        """End-to-end: the guard above must keep `assemble` safe."""
        assembler = WindowAssembler(num_nodes=2, seq_length=4, n_mels=settings.N_MELS)
        good = _window(0, 100.0, seq=4)
        bad = _window(1, 100.0, seq=4)
        bad.timespans = np.full(9, 0.5, dtype=np.float32)

        assembler.push(good)
        assembler.push(bad)
        # Only the well-formed node was buffered, so this cannot raise.
        _, timespans, snapshot = assembler.assemble()
        assert set(snapshot) == {0}
        assert timespans.shape == (1, 4)


class TestDegradedArray:
    """
    One dead microphone must not silence the array.

    The assembler previously required every node with no timeout and no
    eviction, so a single dropout meant the healthy microphones streamed
    indefinitely and nothing was ever emitted. The dashboard then showed
    "waiting for model predictions", which is visually identical to a healthy,
    quiet plant — the worst available failure mode for a monitoring system.
    """

    def _assembler(self, **kwargs):
        defaults = {
            "num_nodes": 4,
            "seq_length": 8,
            "n_mels": settings.N_MELS,
            "staleness_tolerance": 5.0,
            "max_wait": 1.0,
            "min_nodes": 2,
        }
        return WindowAssembler(**{**defaults, **kwargs})

    def test_dead_microphone_releases_a_degraded_snapshot(self):
        assembler = self._assembler()
        for node in range(3):
            assembler.push(_window(node, 100.0))

        # Inside the grace period the array still waits for the fourth mic.
        assert not assembler.is_complete()

        # Past it, three healthy microphones are worth more than silence.
        assert assembler.is_complete(now=time.monotonic() + 5.0)

    def test_degraded_snapshot_zero_fills_and_omits_the_absent_node(self):
        assembler = self._assembler()
        for node in range(3):
            assembler.push(_window(node, 100.0))
        assert assembler.is_complete(now=time.monotonic() + 5.0)

        x, timespans, snapshot = assembler.assemble()

        # Fixed tensor shape: the topology is indexed by position, so a missing
        # node has to keep its slot or every edge after it misaligns.
        assert x.shape == (1, 8, 4 * settings.N_MELS)
        assert timespans.shape == (1, 8)
        mels = settings.N_MELS
        assert torch.allclose(x[0, :, 3 * mels : 4 * mels], torch.zeros(8, mels))

        # ...but no telemetry is fabricated for a microphone that said nothing.
        assert set(snapshot) == {0, 1, 2}
        assert assembler.last_missing == {3}

    def test_stale_node_is_evicted_so_the_array_recovers(self):
        """
        A stale window must be dropped, not retained. Left in place, the spread
        between newest and oldest never falls back inside tolerance, so even the
        surviving microphones stop producing snapshots — a permanent stall that
        no metric reported.
        """
        assembler = self._assembler(num_nodes=2, min_nodes=1)
        assembler.push(_window(0, 100.0))
        assembler.push(_window(1, 100.0))
        assert assembler.is_complete()
        assembler.clear()

        # Node 1 dies; node 0 keeps reporting well past the staleness bound.
        assembler.push(_window(1, 200.0))
        for step in range(5):
            assembler.push(_window(0, 260.0 + step))

        # The stale window is dropped rather than blocking forever, so the
        # surviving microphone can still produce a snapshot.
        assert assembler.is_complete(now=time.monotonic() + 5.0)
        assert 1 not in assembler._windows
        assert set(assembler.assemble()[2]) == {0}

    def test_below_quorum_emits_nothing(self):
        """A graph too sparse to convolve is worse than no answer."""
        assembler = self._assembler(min_nodes=3)
        for node in range(2):
            assembler.push(_window(node, 100.0))
        assert not assembler.is_complete(now=time.monotonic() + 60.0)

    def test_window_of_the_wrong_length_is_rejected_not_raised(self):
        """
        A SEQ_LENGTH mismatch between ingestion and the worker used to reach
        assemble() and raise on the reshape, once per snapshot, forever. It is
        a configuration error, so it is reported once per node and dropped.
        """
        assembler = self._assembler()
        assert assembler.push(_window(0, 100.0, seq=8)) is True
        assert assembler.push(_window(1, 100.0, seq=7)) is False
        assert 1 not in assembler._windows


class TestDecodeWindow:
    def test_roundtrip_from_producer_format(self):
        import msgpack

        seq, mels, frames = 6, settings.N_MELS, 5
        window = np.random.randn(seq, mels, frames).astype(np.float32)
        timespans = np.full(seq, 0.5, dtype=np.float32)

        raw = msgpack.packb(
            {
                "node_id": 2,
                "timestamp": 123.0,
                "window_shape": [seq, mels, frames],
                "window": window.tobytes(),
                "timespans": timespans.tobytes(),
            },
            use_bin_type=True,
        )

        decoded = decode_window(raw)
        assert decoded is not None
        assert decoded.node_id == 2
        # The intra-chunk time axis is averaged away to give one feature vector
        # per timestep, which is the contract the ST-GNN expects.
        assert decoded.features.shape == (seq, mels)
        assert decoded.latest_frame.shape == (mels, frames)
        assert np.allclose(decoded.features, window.mean(axis=2), atol=1e-5)

    def test_garbage_returns_none(self):
        assert decode_window(b"not msgpack at all") is None

    def test_non_finite_rejected(self):
        import msgpack

        seq, mels, frames = 4, settings.N_MELS, 3
        window = np.full((seq, mels, frames), np.nan, dtype=np.float32)
        raw = msgpack.packb(
            {
                "node_id": 0,
                "timestamp": 1.0,
                "window_shape": [seq, mels, frames],
                "window": window.tobytes(),
                "timespans": np.ones(seq, dtype=np.float32).tobytes(),
            },
            use_bin_type=True,
        )
        assert decode_window(raw) is None


@pytest.fixture
def worker(api_client, override_settings):
    override_settings(SEQ_LENGTH=8)
    w = InferenceWorker(
        inference_url="http://testserver", http_client=api_client, load_weights=False
    )
    w.assembler = WindowAssembler(
        num_nodes=settings.NUM_NODES, seq_length=8, n_mels=settings.N_MELS
    )
    yield w


class TestInferenceWorker:
    def test_partial_array_produces_nothing(self, worker):
        for node in range(settings.NUM_NODES - 1):
            assert worker.handle_window(_window(node, time.time())) == []

    def test_complete_array_emits_one_payload_per_node(self, worker):
        now = time.time()
        payloads: list[dict] = []
        for node in range(settings.NUM_NODES):
            payloads = worker.handle_window(_window(node, now)) or payloads

        assert len(payloads) == settings.NUM_NODES
        for payload in payloads:
            assert len(payload["gnn_embedding"]) == settings.GNN_EMBEDDING_DIM
            assert 0.0 <= payload["anomaly_score"] <= 1.0
            assert 0.0 <= payload["degradation_score"] <= 1.0
            assert payload["anomaly_severity"] in {"normal", "warning", "critical"}

    def test_detection_happens_in_the_worker(self, worker):
        """
        Scores must be computed here from model output. The API previously
        accepted them from its caller and forwarded them to Prometheus, so
        nothing in the system actually detected anything.
        """
        worker.scorer.warmup_frames = 5
        worker.scorer.z_threshold = 3.0

        for step in range(20):
            now = time.time() + step
            for node in range(settings.NUM_NODES):
                worker.handle_window(_window(node, now, amplitude=0.05))

        payloads: list[dict] = []
        final = time.time() + 100
        for node in range(settings.NUM_NODES):
            amplitude = 40.0 if node == 1 else 0.05
            result = worker.handle_window(_window(node, final, amplitude=amplitude))
            payloads = result or payloads

        by_node = {p["node_id"]: p for p in payloads}
        assert by_node[1]["is_anomaly"] is True
        assert by_node[1]["z_score"] > by_node[0]["z_score"]

    def test_payloads_are_accepted_by_the_api(self, worker, api_client):
        """The worker's output must satisfy the API's schema."""
        now = time.time()
        payloads: list[dict] = []
        for node in range(settings.NUM_NODES):
            payloads = worker.handle_window(_window(node, now)) or payloads

        for payload in payloads:
            assert api_client.post("/generate_telemetry", json=payload).status_code == 200

    def test_forecast_and_embedding_are_per_node(self, worker):
        """
        Each card on the dashboard must describe its own machine.

        A single facility-level TTF and graph embedding were previously computed
        from the pooled graph readout and copied into every node's payload, so
        the per-node cards were identical by construction. An operator reading
        four "Node N — 47.3%" tiles was reading one number four times.
        """
        now = time.time()
        payloads: list[dict] = []
        for node in range(settings.NUM_NODES):
            # Give each microphone a genuinely different acoustic picture.
            amplitude = 0.05 * (node + 1) ** 3
            payloads = worker.handle_window(_window(node, now, amplitude=amplitude)) or payloads

        assert len(payloads) == settings.NUM_NODES

        embeddings = {tuple(p["gnn_embedding"]) for p in payloads}
        assert len(embeddings) == settings.NUM_NODES, "every node must get its own embedding"

        forecasts = {p["degradation_score"] for p in payloads}
        assert len(forecasts) > 1, "forecasts must not be one pooled number copied N times"

    def test_degraded_array_still_emits_telemetry(self, worker):
        """
        End to end: a dead microphone must not stop the other three.

        ``max_wait=0`` releases as soon as the quorum is met, which keeps the
        test deterministic without sleeping; the quorum is set to N-1 so the
        snapshot fires on exactly the three surviving microphones.
        """
        worker.assembler = WindowAssembler(
            num_nodes=settings.NUM_NODES,
            seq_length=8,
            n_mels=settings.N_MELS,
            max_wait=0.0,
            min_nodes=settings.NUM_NODES - 1,
        )

        now = time.time()
        payloads: list[dict] = []
        for node in range(settings.NUM_NODES - 1):
            payloads = worker.handle_window(_window(node, now)) or payloads

        reporting = {p["node_id"] for p in payloads}
        assert reporting == set(range(settings.NUM_NODES - 1))
        assert settings.NUM_NODES - 1 not in reporting

    def test_flagged_frames_carry_spectral_evidence(self, worker):
        """
        "Node 3, anomaly score 0.87" is not actionable. A technician needs to
        know what the system heard, and which catalogued fault that resembles.
        """
        # The autoencoder is resident (if untrained) — enough for the error map
        # to decompose, which is all the attribution needs.
        worker.weights_loaded = True
        worker.scorer.warmup_frames = 3

        payloads: list[dict] = []
        for step in range(8):
            now = time.time() + step
            for node in range(settings.NUM_NODES):
                worker.handle_window(_window(node, now, amplitude=0.05))

        final = time.time() + 100
        for node in range(settings.NUM_NODES):
            amplitude = 60.0 if node == 1 else 0.05
            payloads = worker.handle_window(_window(node, final, amplitude=amplitude)) or payloads

        flagged = [p for p in payloads if p["is_anomaly"]]
        assert flagged, "expected the loud node to flag"
        for payload in flagged:
            assert payload["explanation"]["bands"]
            assert payload["explanation"]["summary"]
            assert payload["diagnosis"]["fault"]
            assert payload["diagnosis"]["urgency"] in {"monitor", "schedule", "urgent"}

    def test_quiet_frames_pay_nothing_for_attribution(self, worker):
        """Attribution is off the steady-state path; most frames are normal."""
        worker.weights_loaded = True
        now = time.time()
        payloads: list[dict] = []
        for node in range(settings.NUM_NODES):
            payloads = worker.handle_window(_window(node, now, amplitude=0.05)) or payloads

        assert payloads
        assert all(not p["is_anomaly"] for p in payloads)
        assert all("explanation" not in p for p in payloads)

    def test_untrained_worker_does_not_invent_an_explanation(self, worker):
        """
        Without a trained autoencoder the scorer falls back to frame energy, so
        there is no reconstruction-error map to decompose and any attribution
        would be fabricated.
        """
        assert worker.weights_loaded is False
        worker.scorer.warmup_frames = 1

        payloads: list[dict] = []
        for step in range(4):
            now = time.time() + step
            for node in range(settings.NUM_NODES):
                amplitude = 60.0 if (node == 1 and step == 3) else 0.05
                result = worker.handle_window(_window(node, now, amplitude=amplitude))
                payloads = result or payloads

        assert all("explanation" not in p for p in payloads)

    def test_no_sinks_configured_means_nobody_is_paged(self, worker):
        """
        An unconfigured deployment must route nowhere. Paging somebody because a
        URL was lying around in the environment is worse than staying quiet.
        """
        assert worker.alerts.sinks == []

        now = time.time()
        for node in range(settings.NUM_NODES):
            worker.handle_window(_window(node, now))  # must not raise

    def _capture_alerts(self, worker) -> list[dict]:
        """Point the worker's router at an in-memory sink."""
        from src.alerting.webhook import GenericWebhookSink

        delivered: list[dict] = []
        worker.alerts.sinks = [
            GenericWebhookSink(
                "https://example.invalid",
                transport=lambda url, payload, headers: delivered.append(payload) or True,
            )
        ]
        return delivered

    @staticmethod
    def _scored(severity: str, *, node_id: int = 1, fault: str | None = "Bearing race defect"):
        payload = {
            "node_id": node_id,
            "timestamp": 1000.0,
            "anomaly_score": 0.9,
            "anomaly_severity": severity,
            "degradation_score": 0.4,
            "is_anomaly": severity != "normal",
            "z_score": 6.0,
        }
        if fault is not None:
            payload["diagnosis"] = {
                "fault": fault,
                "confidence": 0.7,
                "recommended_action": "Inspect the bearing.",
                "evidence": ["2.1-3.4 kHz (46%)"],
            }
        return payload

    def test_alerts_reach_configured_sinks(self, worker):
        """A prediction nobody sees changes nothing."""
        delivered = self._capture_alerts(worker)

        worker._raise_alert(self._scored("critical"))

        assert delivered, "a flagged frame must reach the configured sink"
        assert delivered[0]["node_id"] == 1
        assert delivered[0]["fault"] == "Bearing race defect"

    def test_a_healthy_frame_pages_nobody(self, worker):
        delivered = self._capture_alerts(worker)

        worker._raise_alert(self._scored("normal"))

        assert delivered == []

    def test_a_steady_fault_does_not_re_page_every_frame(self, worker):
        """
        Cooldown is what keeps an alerting integration from being muted. The
        pipeline scores every node twice a second; a sustained fault that paged
        on each one would be silenced by a human within a shift.
        """
        delivered = self._capture_alerts(worker)

        for _ in range(20):
            worker._raise_alert(self._scored("critical"))

        assert len(delivered) == 1, f"cooldown did not suppress; sent {len(delivered)}"
        assert worker.alerts.suppressed_count == 19

    def test_escalation_breaks_through_the_cooldown(self, worker):
        """A fault getting worse is new information, not a repeat."""
        delivered = self._capture_alerts(worker)

        worker._raise_alert(self._scored("warning"))
        worker._raise_alert(self._scored("critical"))

        assert [d["severity"] for d in delivered] == ["warning", "critical"]

    def test_recovery_clears_the_alert(self, worker):
        delivered = self._capture_alerts(worker)

        worker._raise_alert(self._scored("critical"))
        worker._raise_alert(self._scored("normal"))

        assert [d["resolved"] for d in delivered] == [False, True]

    def test_an_undiagnosed_anomaly_still_pages(self, worker):
        """
        An unrecognised signature on a machine that was quiet yesterday still
        deserves a look; dropping it because the catalogue had no match would
        lose exactly the novel faults worth knowing about.
        """
        delivered = self._capture_alerts(worker)

        worker._raise_alert(self._scored("critical", fault=None))

        assert delivered
        assert delivered[0]["fault"] == "Unrecognised acoustic anomaly"

    def test_a_wedged_webhook_does_not_stop_scoring(self, worker):
        """Alerting is downstream of safety; it must never break the pipeline."""
        from src.alerting.webhook import GenericWebhookSink

        def exploding_transport(url, payload, headers):
            raise RuntimeError("webhook is down")

        worker.alerts.sinks = [
            GenericWebhookSink("https://example.invalid", transport=exploding_transport)
        ]
        worker.alerts.min_severity = "normal"

        now = time.time()
        payloads: list[dict] = []
        for node in range(settings.NUM_NODES):
            payloads = worker.handle_window(_window(node, now)) or payloads

        assert len(payloads) == settings.NUM_NODES

    def test_submit_survives_unreachable_api(self):
        w = InferenceWorker(inference_url="http://127.0.0.1:1", load_weights=False)
        try:
            assert w.submit({"node_id": 0}) is False
        finally:
            w.close()


def _scripted_worker(statuses: list[int], override_settings, dead_letter=None):
    """A worker whose API answers with ``statuses`` in turn (the last one repeats)."""
    import httpx

    override_settings(SEQ_LENGTH=8, TELEMETRY_RETRY_BACKOFF=0.0, TELEMETRY_MAX_RETRIES=2)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses[min(len(calls), len(statuses) - 1)]
        calls.append(status)
        return httpx.Response(status, json={})

    w = InferenceWorker(
        inference_url="http://testserver",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        load_weights=False,
        dead_letter=dead_letter,
    )
    w.assembler = WindowAssembler(
        num_nodes=settings.NUM_NODES, seq_length=8, n_mels=settings.N_MELS
    )
    return w, calls


def _emit_snapshot(worker: InferenceWorker) -> list[dict]:
    now = time.time()
    payloads: list[dict] = []
    for node in range(settings.NUM_NODES):
        payloads = worker.handle_window(_window(node, now)) or payloads
    return payloads


class TestTelemetryDelivery:
    """
    Failed submissions used to be counted as dropped while the loop committed
    offsets regardless, so a scored result the API never accepted existed
    nowhere. They are now retried when the failure is transient and otherwise
    kept on a dead-letter topic with the reason.
    """

    def test_transient_failure_is_retried_then_succeeds(self, override_settings):
        w, calls = _scripted_worker([503, 503, 200], override_settings)
        assert w.deliver({"node_id": 0}) is None
        assert calls == [503, 503, 200]

    def test_exhausted_retries_report_the_reason(self, override_settings):
        w, calls = _scripted_worker([503], override_settings)
        assert w.deliver({"node_id": 0}) == "server_error"
        assert len(calls) == 1 + settings.TELEMETRY_MAX_RETRIES

    def test_rejected_payload_is_not_retried(self, override_settings):
        """A 4xx will be rejected however many times it is sent."""
        w, calls = _scripted_worker([422], override_settings)
        assert w.deliver({"node_id": 0}) == "rejected"
        assert len(calls) == 1

    def test_throttling_is_not_retried(self, override_settings):
        """The limiter window is 60 s; a sub-second retry cannot outwait it."""
        w, calls = _scripted_worker([429], override_settings)
        assert w.deliver({"node_id": 0}) == "throttled"
        assert len(calls) == 1

    def test_undelivered_payloads_are_dead_lettered(self, override_settings):
        kept: list[tuple[dict, str]] = []
        w, _ = _scripted_worker(
            [500], override_settings, dead_letter=lambda p, r: kept.append((p, r))
        )
        payloads = _emit_snapshot(w)
        assert len(kept) == len(payloads) == settings.NUM_NODES
        assert {reason for _, reason in kept} == {"server_error"}
        assert [p["node_id"] for p, _ in kept] == [p["node_id"] for p in payloads]

    def test_an_outage_does_not_retry_every_payload(self, override_settings):
        """After one payload exhausts its retries, the rest of the snapshot skips them."""
        w, calls = _scripted_worker([500], override_settings, dead_letter=lambda p, r: None)
        _emit_snapshot(w)
        assert len(calls) == (1 + settings.TELEMETRY_MAX_RETRIES) + (settings.NUM_NODES - 1)

    def test_delivered_payloads_are_not_dead_lettered(self, override_settings):
        kept: list = []
        w, _ = _scripted_worker([200], override_settings, dead_letter=lambda p, r: kept.append(p))
        assert len(_emit_snapshot(w)) == settings.NUM_NODES
        assert kept == []

    def test_a_failing_dead_letter_sink_does_not_stop_scoring(self, override_settings):
        def broken(_payload, _reason):
            raise RuntimeError("broker down")

        w, _ = _scripted_worker([500], override_settings, dead_letter=broken)
        assert len(_emit_snapshot(w)) == settings.NUM_NODES


class TestDeadLetterWriter:
    class _Producer:
        def __init__(self, fail: bool = False, remaining: int = 0):
            self.fail, self.remaining, self.sent, self._pending = fail, remaining, [], []

        def produce(self, topic, key=None, value=None, on_delivery=None):
            self.sent.append((topic, key, value))
            self._pending.append(on_delivery)

        def poll(self, _timeout):
            return 0

        def flush(self, _timeout):
            for cb in self._pending:
                cb("delivery failed" if self.fail else None, None)
            self._pending.clear()
            return self.remaining

    def test_record_carries_payload_and_reason(self):
        import json

        from src.inference.worker import _DeadLetterWriter

        producer = self._Producer()
        writer = _DeadLetterWriter(producer)
        writer({"node_id": 2, "anomaly_score": 0.9}, "unreachable")
        assert writer.settle() is True

        topic, key, value = producer.sent[0]
        record = json.loads(value)
        assert topic == settings.TELEMETRY_DLQ_TOPIC
        assert key == b"2"
        assert record["reason"] == "unreachable"
        assert record["payload"]["anomaly_score"] == 0.9

    def test_unacknowledged_writes_do_not_settle(self):
        """The worker must not commit offsets for payloads that exist nowhere."""
        from src.inference.worker import _DeadLetterWriter

        failing = _DeadLetterWriter(self._Producer(fail=True))
        failing({"node_id": 0}, "server_error")
        assert failing.settle() is False

        stuck = _DeadLetterWriter(self._Producer(remaining=1))
        stuck({"node_id": 0}, "server_error")
        assert stuck.settle() is False


class TestControlCommands:
    def _warmed(self, worker: InferenceWorker, node: int = 0) -> None:
        g = torch.Generator().manual_seed(0)
        for _ in range(settings.ANOMALY_WARMUP_FRAMES + 10):
            worker.scorer.score(node, torch.rand(settings.N_MELS, 6, generator=g))

    def test_rebaseline_reaches_the_scorer(self, worker):
        self._warmed(worker)
        assert worker.apply_control(
            {"command": "rebaseline", "array_id": settings.ARRAY_ID, "node_id": 0, "reason": "x"}
        )
        assert worker.scorer.get_node_summary()[0]["drift_guard_armed"]

    def test_a_command_for_another_array_is_ignored(self, worker):
        assert not worker.apply_control(
            {"command": "rebaseline", "array_id": "some-other-array", "node_id": 0}
        )

    def test_unknown_commands_are_ignored(self, worker):
        assert not worker.apply_control({"command": "format-disk"})

    def test_cli_message_requires_a_reason(self):
        from src.inference.control import build_rebaseline

        with pytest.raises(ValueError, match="reason"):
            build_rebaseline(0, "   ", "tester")
        with pytest.raises(ValueError, match="outside this array"):
            build_rebaseline(settings.NUM_NODES, "valid", "tester")
        command = build_rebaseline(None, "line restarted", "tester")
        assert command["command"] == "rebaseline"
        assert command["node_id"] is None
        assert command["array_id"] == settings.ARRAY_ID


class TestArrayIdentity:
    def _raw(self, **extra) -> bytes:
        import msgpack

        seq, mels, frames = 4, settings.N_MELS, 3
        payload = {
            "node_id": 1,
            "timestamp": 5.0,
            "window_shape": [seq, mels, frames],
            "window": np.zeros((seq, mels, frames), dtype=np.float32).tobytes(),
            "timespans": np.ones(seq, dtype=np.float32).tobytes(),
            **extra,
        }
        return msgpack.packb(payload, use_bin_type=True)

    def test_window_reports_its_array(self):
        assert decode_window(self._raw(array_id="hall-3")).array_id == "hall-3"

    def test_legacy_window_is_attributed_to_this_array(self):
        """Windows produced before array keying can only be from this deployment."""
        assert decode_window(self._raw()).array_id == settings.ARRAY_ID


class TestPayloadCarriesDrift:
    def test_drift_z_is_reported(self, worker):
        for payload in _emit_snapshot(worker):
            assert "drift_z" in payload
