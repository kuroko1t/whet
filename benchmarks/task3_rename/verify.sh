#!/bin/sh
set -e

# No remaining references to the old name.
if grep -rn '\bcompute\b' . >/dev/null 2>&1; then
    echo "FAIL: 'compute' still referenced:"
    grep -rn '\bcompute\b' .
    exit 1
fi

# `double` must remain (we did not ask to rename it).
grep -q '\bdouble\b' calc.py || { echo "FAIL: double removed from calc.py"; exit 1; }
grep -q '\bdouble\b' main.py || { echo "FAIL: double removed from main.py"; exit 1; }

# Behaviour intact via the new name.
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("calc", "calc.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert hasattr(m, "add"), "function not renamed to add"
assert m.add(2, 3) == 5, f"add(2,3) = {m.add(2,3)}"
assert m.double(4) == 8, f"double(4) = {m.double(4)}"
PY

# End-to-end scripts run cleanly.
python test_calc.py >/dev/null
python main.py >/dev/null
echo OK
