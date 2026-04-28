#!/bin/sh
set -e
python - <<'PY'
import sys, importlib.util
spec = importlib.util.spec_from_file_location("hello", "hello.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert hasattr(m, "greet"), "greet was removed"
assert m.greet("World") == "Hello, World!", "greet behaviour changed"
assert hasattr(m, "farewell"), "farewell was not added"
assert m.farewell("World") == "Goodbye, World!", f"farewell returned {m.farewell('World')!r}"
print("OK")
PY
