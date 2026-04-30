"""Billing worker.

This worker pulls jobs off the billing queue and routes them through the
event bus. The string `process_event` appears in this docstring as a
reference but the only real call is in `_drain`."""

from eventbus import process_event


def _drain(queue):
    while True:
        job = queue.pop()
        if job is None:
            return
        event = {"kind": "billing", "job": job}
        process_event(event)


def health_check():
    return "ok"
