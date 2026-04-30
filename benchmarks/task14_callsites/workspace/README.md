# Sample Service

A small Python service organised around an event bus.

## Layout
- `eventbus.py` — defines `process_event`, the central dispatcher.
- `handlers/` — request-time handlers (login, payment, notification, admin).
- `workers/` — background workers (email, billing, retry, cleanup).
- `utils/` — pure helpers (timing, strings, metrics).
- `cli.py` — entry point.

## Notes
Some files mention `process_event` in comments or docstrings without
actually calling it. Don't conflate references with real calls.
