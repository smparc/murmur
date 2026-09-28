"""
Operator commands to the inference worker.

    murmur-rebaseline --node 2 --reason "pump P-201 re-rated to 1450 rpm"
    murmur-rebaseline --all --reason "line restarted after annual shutdown"

The anomaly scorer never absorbs a persistent departure on its own: from one
microphone's score alone, a legitimate change of operating point and a fault
that has stopped getting worse look identical, so treating either as the other
must be a decision somebody makes. This is how that decision reaches the
worker — a command on the control topic, keyed by array so it lands on the
worker that owns the array.

It needs broker access, not the API key: re-baselining silences an active
detection, so it belongs to whoever operates the pipeline, not to anyone who can
load the dashboard. Every command is logged by the worker with who asked and why.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import time

from src.settings import settings


def build_rebaseline(node_id: int | None, reason: str, requested_by: str) -> dict:
    """The control message the worker's ``apply_control`` understands."""
    if node_id is not None and not 0 <= node_id < settings.NUM_NODES:
        raise ValueError(f"node {node_id} is outside this array (0..{settings.NUM_NODES - 1})")
    if not reason.strip():
        raise ValueError("a reason is required; it is the audit trail for silencing a detection")
    return {
        "command": "rebaseline",
        "array_id": settings.ARRAY_ID,
        "node_id": node_id,
        "reason": reason.strip(),
        "requested_by": requested_by,
        "issued_at": time.time(),
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a broker
    parser = argparse.ArgumentParser(
        description="Accept a node's current sound as its new normal baseline."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--node", type=int, help="microphone to re-baseline")
    target.add_argument("--all", action="store_true", help="re-baseline every microphone")
    parser.add_argument("--reason", required=True, help="why this change is legitimate")
    parser.add_argument("--requested-by", default=getpass.getuser())
    args = parser.parse_args(argv)

    try:
        command = build_rebaseline(None if args.all else args.node, args.reason, args.requested_by)
    except ValueError as exc:
        parser.error(str(exc))

    from confluent_kafka import Producer

    producer = Producer({"bootstrap.servers": settings.KAFKA_BROKER, "acks": "all"})
    errors: list[str] = []
    producer.produce(
        settings.CONTROL_TOPIC,
        key=settings.ARRAY_ID.encode("utf-8"),
        value=json.dumps(command).encode("utf-8"),
        on_delivery=lambda err, _msg: errors.append(str(err)) if err else None,
    )
    if producer.flush(10.0) or errors:
        print(f"Control command not delivered: {errors or 'timed out'}", file=sys.stderr)
        return 1
    print(f"Sent to {settings.CONTROL_TOPIC}: {json.dumps(command)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
