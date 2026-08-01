"""
Alert delivery.

A prediction nobody sees changes nothing. This package routes faults to Slack,
PagerDuty or a generic webhook — and, more importantly, decides when *not* to.
See :mod:`src.alerting.webhook` for why suppression is the hard part.
"""

from src.alerting.webhook import (
    Alert,
    AlertRouter,
    AlertSink,
    GenericWebhookSink,
    PagerDutySink,
    SlackSink,
    urllib_transport,
)

__all__ = [
    "Alert",
    "AlertRouter",
    "AlertSink",
    "GenericWebhookSink",
    "PagerDutySink",
    "SlackSink",
    "urllib_transport",
]
