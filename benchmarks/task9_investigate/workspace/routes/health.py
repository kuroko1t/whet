"""Health-check route."""

routes = [
    ("GET", "/health", "health"),
]


def health():
    return {"status": "ok"}
