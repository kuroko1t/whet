"""Event bus core. Defines `process_event` — the thing every callsite below
calls into. This is the only file where `process_event` is *defined*."""


def process_event(event):
    """Dispatch a single event to whatever subsystem cares about it."""
    if event is None:
        return None
    return _route(event)


def _route(event):
    return event.get("kind")
