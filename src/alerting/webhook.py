"""
Turning a prediction into something a person acts on.

Murmur can detect a fault, locate it, explain it and name it, and none of that
reaches anyone unless it leaves the process. This module routes alerts to Slack,
PagerDuty or a generic webhook.

The hard part is not the HTTP call — it is *not sending*. A detector scoring one
frame every 500 ms across sixteen microphones has 115,000 opportunities per hour
to page somebody, and a sustained fault is anomalous on every single frame until
it is repaired. An integration that forwards each one is muted by the end of the
first shift, after which the system is decorative. So the router is built around
suppression:

- **Cooldown.** One alert per node and fault, then silence for a configurable
  window regardless of how many frames keep firing.
- **Escalation override.** Severity increasing from warning to critical breaks
  the cooldown, because that is genuinely new information.
- **Resolution notices.** A node returning to normal sends a clearing message,
  so nobody chases a fault that has already stopped.

Transport is injectable and defaults to ``urllib`` from the standard library:
Murmur has enough dependencies, and an alerting path that fails to import is
worse than one that is slightly less ergonomic.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

log = logging.getLogger(__name__)

__all__ = [
    "Alert",
    "AlertRouter",
    "AlertSink",
    "GenericWebhookSink",
    "PagerDutySink",
    "SlackSink",
    "Transport",
    "urllib_transport",
]

Severity = Literal["normal", "warning", "critical"]

#: Severity ordering, for detecting escalation.
_SEVERITY_RANK: dict[str, int] = {"normal": 0, "warning": 1, "critical": 2}


@dataclass(frozen=True)
class Alert:
    """Everything a responder needs, in one payload."""

    node_id: int
    severity: Severity
    fault: str
    confidence: float
    anomaly_score: float = 0.0
    ttf_prediction: float = 0.0
    evidence: tuple[str, ...] = field(default_factory=tuple)
    recommended_action: str = ""
    location: tuple[float, float, float] | None = None
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False

    @property
    def key(self) -> tuple[int, str]:
        """Deduplication key: one ongoing alert per node and fault."""
        return (self.node_id, self.fault)

    @property
    def title(self) -> str:
        if self.resolved:
            return f"RESOLVED: node {self.node_id} back to normal"
        return f"{self.severity.upper()}: {self.fault} on node {self.node_id}"

    def body(self) -> str:
        """Plain-text summary, used by every sink that does not want structure."""
        if self.resolved:
            return f"Node {self.node_id} has returned to baseline. Previous fault: {self.fault}."

        lines = [
            f"Node {self.node_id} — {self.fault} ({self.confidence:.0%} confidence)",
            f"Anomaly score {self.anomaly_score:.3f} | "
            f"Failure probability {self.ttf_prediction:.0%}",
        ]
        if self.evidence:
            lines.append("Evidence: " + "; ".join(self.evidence))
        if self.location is not None:
            x, y, _ = self.location
            lines.append(f"Estimated position: ({x:.1f} m, {y:.1f} m)")
        if self.recommended_action:
            lines.append(f"Action: {self.recommended_action}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "severity": self.severity,
            "fault": self.fault,
            "confidence": round(self.confidence, 4),
            "anomaly_score": round(self.anomaly_score, 4),
            "ttf_prediction": round(self.ttf_prediction, 4),
            "evidence": list(self.evidence),
            "recommended_action": self.recommended_action,
            "location": list(self.location) if self.location else None,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
        }


class Transport(Protocol):
    """Sends a JSON body to a URL. Returns True on success."""

    def __call__(self, url: str, payload: dict, headers: dict[str, str]) -> bool: ...


def urllib_transport(url: str, payload: dict, headers: dict[str, str]) -> bool:
    """Default transport: a blocking POST via the standard library."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", **headers}
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        # Never propagate: a failing alerting integration must not take down the
        # detector that produced the alert.
        log.warning("Alert delivery to %s failed: %s", url, exc)
        return False


class AlertSink:
    """Base sink. Subclasses shape the payload for a particular service."""

    name = "sink"

    def __init__(self, url: str, transport: Transport = urllib_transport):
        if not url:
            raise ValueError("sink requires a URL")
        self.url = url
        self.transport = transport

    def format(self, alert: Alert) -> dict:
        raise NotImplementedError

    def headers(self) -> dict[str, str]:
        return {}

    def send(self, alert: Alert) -> bool:
        return self.transport(self.url, self.format(alert), self.headers())


class SlackSink(AlertSink):
    """Slack incoming webhook, formatted as blocks."""

    name = "slack"

    _COLOURS = {"critical": "#dc2626", "warning": "#f59e0b", "normal": "#10b981"}

    def format(self, alert: Alert) -> dict:
        colour = (
            self._COLOURS["normal"]
            if alert.resolved
            else self._COLOURS.get(alert.severity, "#6b7280")
        )
        return {
            "text": alert.title,
            "attachments": [
                {
                    "color": colour,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {"type": "plain_text", "text": alert.title},
                        },
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": alert.body()},
                        },
                    ],
                }
            ],
        }


class PagerDutySink(AlertSink):
    """
    PagerDuty Events API v2.

    Uses ``dedup_key`` so PagerDuty groups repeats into one incident, and sends
    ``resolve`` when the node recovers — which lets an incident close itself
    rather than waiting for someone to notice it is stale.
    """

    name = "pagerduty"

    _SEVERITY = {"critical": "critical", "warning": "warning", "normal": "info"}

    def __init__(
        self,
        routing_key: str,
        url: str = "https://events.pagerduty.com/v2/enqueue",
        transport: Transport = urllib_transport,
    ):
        super().__init__(url, transport)
        if not routing_key:
            raise ValueError("PagerDuty requires a routing key")
        self.routing_key = routing_key

    def format(self, alert: Alert) -> dict:
        return {
            "routing_key": self.routing_key,
            "event_action": "resolve" if alert.resolved else "trigger",
            "dedup_key": f"murmur-node-{alert.node_id}-{alert.fault}",
            "payload": {
                "summary": alert.title,
                "source": f"murmur-node-{alert.node_id}",
                "severity": self._SEVERITY.get(alert.severity, "warning"),
                "custom_details": alert.as_dict(),
            },
        }


class GenericWebhookSink(AlertSink):
    """Posts the raw alert payload — for a CMMS, work-order system or bridge."""

    name = "webhook"

    def __init__(
        self,
        url: str,
        transport: Transport = urllib_transport,
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(url, transport)
        self.extra_headers = extra_headers or {}

    def format(self, alert: Alert) -> dict:
        return alert.as_dict()

    def headers(self) -> dict[str, str]:
        return self.extra_headers


class AlertRouter:
    """
    Fans alerts out to sinks, with the suppression that makes them survivable.

    Parameters
    ----------
    sinks:
        Where to deliver. An empty list makes the router a no-op, which is the
        right default for a system that should not page anyone until someone
        has deliberately configured it to.
    cooldown_s:
        Silence for a given node and fault after an alert is sent.
    min_severity:
        Alerts below this are dropped. Defaults to ``warning`` — ``normal`` is
        every healthy frame.
    notify_on_resolve:
        Send a clearing message when a node returns to normal.
    """

    def __init__(
        self,
        sinks: list[AlertSink] | None = None,
        cooldown_s: float = 900.0,
        min_severity: Severity = "warning",
        notify_on_resolve: bool = True,
        clock: Callable[[], float] = time.time,
    ):
        if cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")
        if min_severity not in _SEVERITY_RANK:
            raise ValueError(f"min_severity must be one of {sorted(_SEVERITY_RANK)}")

        self.sinks = sinks or []
        self.cooldown_s = cooldown_s
        self.min_severity = min_severity
        self.notify_on_resolve = notify_on_resolve
        self._clock = clock
        self._last_sent: dict[tuple[int, str], float] = {}
        self._last_severity: dict[tuple[int, str], str] = {}
        self.suppressed_count = 0

    def _should_send(self, alert: Alert) -> bool:
        if _SEVERITY_RANK[alert.severity] < _SEVERITY_RANK[self.min_severity]:
            return False

        previous_severity = self._last_severity.get(alert.key)
        escalated = previous_severity is not None and (
            _SEVERITY_RANK[alert.severity] > _SEVERITY_RANK[previous_severity]
        )
        if escalated:
            # Getting worse is new information; the cooldown does not apply.
            return True

        last = self._last_sent.get(alert.key)
        return last is None or (self._clock() - last) >= self.cooldown_s

    def send(self, alert: Alert) -> bool:
        """
        Deliver an alert unless it is suppressed.

        Returns True when it was delivered to at least one sink.
        """
        if alert.resolved:
            if not self.notify_on_resolve or alert.key not in self._last_sent:
                # Never alerted on this fault, so there is nothing to clear.
                return False
            delivered = self._dispatch(alert)
            self._last_sent.pop(alert.key, None)
            self._last_severity.pop(alert.key, None)
            return delivered

        if not self._should_send(alert):
            self.suppressed_count += 1
            return False

        delivered = self._dispatch(alert)
        # Record the attempt either way. Retrying a failing webhook on every
        # frame would turn an outage into a flood.
        self._last_sent[alert.key] = self._clock()
        self._last_severity[alert.key] = alert.severity
        return delivered

    def _dispatch(self, alert: Alert) -> bool:
        delivered = False
        for sink in self.sinks:
            try:
                if sink.send(alert):
                    delivered = True
                else:
                    log.warning("Sink %s did not accept alert %s", sink.name, alert.key)
            except Exception:
                log.exception("Sink %s raised while sending %s", sink.name, alert.key)
        return delivered

    def reset(self, node_id: int | None = None) -> None:
        """Clear suppression state, for one node or all of them."""
        if node_id is None:
            self._last_sent.clear()
            self._last_severity.clear()
            return
        for key in [k for k in self._last_sent if k[0] == node_id]:
            self._last_sent.pop(key, None)
            self._last_severity.pop(key, None)
