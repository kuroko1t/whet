#!/bin/sh
# Verify task12_test_gen:
#   1. calculator.py must be unchanged (anti-tampering).
#   2. test_calculator.py must define at least 5 test functions and they must pass.
#   3. The suite must catch a deliberate mutation we apply to a private copy
#      of the calculator (mutation testing — proves the tests aren't trivial).
set -e

# 1. Pin calculator.py.
expected_calc_sha="d621553bf802ea45541f4d21f4ba96ee372a086a7232466a79044f9d67ec9a77"
actual_calc_sha=$(sha256sum calculator.py | awk '{print $1}')
if [ "$actual_calc_sha" != "$expected_calc_sha" ]; then
    echo "FAIL: calculator.py was modified."
    echo "  expected: $expected_calc_sha"
    echo "  actual:   $actual_calc_sha"
    exit 1
fi

# 2. Test count and pass.
collected=$(python -m pytest --collect-only -q test_calculator.py 2>/dev/null \
            | grep -cE '::test_[A-Za-z_0-9]+' || true)
if [ "${collected:-0}" -lt 5 ]; then
    echo "FAIL: expected >=5 collected tests, got $collected"
    python -m pytest --collect-only -q test_calculator.py 2>&1 | tail -10
    exit 1
fi

python -m pytest -q test_calculator.py

# 3. Mutation testing. Replace calculator.py with a buggy variant and run the
# user's tests again — they MUST fail. We swap divide(a, b) to compute a + b
# (instead of a / b), which is a behaviourally distinct bug any reasonable
# divide test should catch.
TMP_BACKUP=$(mktemp)
cp calculator.py "$TMP_BACKUP"
python - <<'PY'
import pathlib, re
src = pathlib.Path("calculator.py").read_text()
# Replace `return a / b` inside divide() with `return a + b`.
mutated = re.sub(
    r"def divide\(self, a, b\):.*?return a / b",
    """def divide(self, a, b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("inputs must be numeric")
        if b == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return a + b""",
    src,
    count=1,
    flags=re.DOTALL,
)
if mutated == src:
    raise SystemExit("internal: failed to apply mutation")
pathlib.Path("calculator.py").write_text(mutated)
PY

set +e
python -m pytest -q test_calculator.py >/dev/null 2>&1
mutation_rc=$?
set -e

# Restore the canonical calculator.
cp "$TMP_BACKUP" calculator.py

if [ "$mutation_rc" -eq 0 ]; then
    echo "FAIL: tests still pass against a mutated divide() — your test suite does not actually exercise division correctness."
    exit 1
fi

echo OK
