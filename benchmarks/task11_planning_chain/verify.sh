#!/bin/sh
# Verify task11_planning_chain:
#   1. test_pipeline.py must be untouched (anti-tampering).
#   2. No `print(` remains in main.py, worker.py, utils.py.
#   3. Each source file imports `logging` and obtains a logger.
#   4. pytest passes.
set -e

# 1. Pin the test file.
expected_test_sha="842fced1f1c84099a2dc8cf2560d537fbec3c53bb6a0d14bf318b7a66c03178d"
actual_test_sha=$(sha256sum test_pipeline.py | awk '{print $1}')
if [ "$actual_test_sha" != "$expected_test_sha" ]; then
    echo "FAIL: test_pipeline.py was modified."
    echo "  expected: $expected_test_sha"
    echo "  actual:   $actual_test_sha"
    exit 1
fi

# 2. No more print() calls in source files.
for f in main.py worker.py utils.py; do
    if grep -nE '\bprint\s*\(' "$f"; then
        echo "FAIL: $f still calls print()"
        exit 1
    fi
done

# 3. Each source file must import logging and create a logger.
for f in main.py worker.py utils.py; do
    grep -q '^import logging' "$f" || {
        echo "FAIL: $f does not import logging"
        exit 1
    }
    grep -qE 'logging\.getLogger\(' "$f" || {
        echo "FAIL: $f does not call logging.getLogger(...)"
        exit 1
    }
done

# 4. pytest must pass.
python -m pytest -q test_pipeline.py
echo OK
