#!/bin/sh
# Pass if no "recieve"/"Recieve" remains anywhere and at least 5 "receive"/"Receive" exist.
set -e

remaining=$(grep -rEi 'recieve' . 2>/dev/null | wc -l)
if [ "$remaining" -ne 0 ]; then
    echo "FAIL: 'recieve' still present in $remaining line(s)"
    grep -rEni 'recieve' . 2>/dev/null
    exit 1
fi

found=$(grep -rEci 'receive' . 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
if [ "$found" -lt 5 ]; then
    echo "FAIL: expected >=5 occurrences of 'receive', found $found"
    exit 1
fi

# server.py must still parse and the renamed function must work.
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("server", "server.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert hasattr(m, "receive_request"), "function should be renamed to receive_request"
assert not hasattr(m, "recieve_request"), "old misspelled function still defined"
assert m.handle("ping") == "ping", f"handle behaviour changed: {m.handle('ping')!r}"
print("OK")
PY
