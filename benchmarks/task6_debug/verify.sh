#!/bin/sh
# Verify task6_debug:
#   1. test_buggy.py must be unchanged (no test tampering).
#   2. All tests in test_buggy.py must pass.
# Runs in the workspace copy directory.
set -e

EXPECTED_TEST_SHA="acd88745ceb69162330a8c23cece32d93f65d56b4f7d558e1bef8c9224ed525c"
ACTUAL_TEST_SHA=$(sha256sum test_buggy.py | awk '{print $1}')
if [ "$ACTUAL_TEST_SHA" != "$EXPECTED_TEST_SHA" ]; then
    echo "FAIL: test_buggy.py was modified by the agent."
    echo "  expected SHA: $EXPECTED_TEST_SHA"
    echo "  actual SHA:   $ACTUAL_TEST_SHA"
    exit 1
fi

python test_buggy.py
