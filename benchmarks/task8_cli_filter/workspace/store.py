"""JSON-file persistence for the tasks CLI."""
import json
from pathlib import Path

DEFAULT_STORE = Path.home() / ".tasks.json"


def load(path=None):
    p = Path(path) if path else DEFAULT_STORE
    if not p.exists():
        return []
    return json.loads(p.read_text())


def save(tasks, path=None):
    p = Path(path) if path else DEFAULT_STORE
    p.write_text(json.dumps(tasks, indent=2))
