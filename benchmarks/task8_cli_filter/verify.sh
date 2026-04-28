#!/bin/sh
# Verify task8_cli_filter:
#   1. Existing four test functions must still be present (anti-tampering by name).
#   2. pytest passes with at least 5 tests (4 existing + >= 1 new).
#   3. External behavioural test of the --status filter (independent of the model's tests).
set -e

# 1. Existing tests must still be defined verbatim by name.
for t in test_add_creates_task test_list_shows_tasks test_done_marks_task test_remove_deletes_task; do
    if ! grep -q "def $t" test_cli.py; then
        echo "FAIL: existing test '$t' was removed from test_cli.py"
        exit 1
    fi
done

# 2. Test count must have grown (collect-only count >= 5).
collected=$(python -m pytest --collect-only -q test_cli.py 2>/dev/null | grep -cE '::test_[A-Za-z_0-9]+' || true)
if [ "${collected:-0}" -lt 5 ]; then
    echo "FAIL: expected >=5 collected tests (4 existing + new), got $collected"
    python -m pytest --collect-only -q test_cli.py 2>&1 | tail -10
    exit 1
fi

python -m pytest -q test_cli.py

# 3. External behavioural test (cannot be gamed by editing test_cli.py).
TMP=$(mktemp -d -t tasks-verify-XXXXXX)
STORE="$TMP/tasks.json"
py() { python cli.py --store "$STORE" "$@"; }

py add "alpha"  >/dev/null
py add "beta"   >/dev/null
py add "gamma"  >/dev/null
py done 2       >/dev/null

# 3a. No filter → all three.
out=$(py list)
echo "$out" | grep -q alpha || { echo "FAIL: alpha missing in unfiltered list"; exit 1; }
echo "$out" | grep -q beta  || { echo "FAIL: beta missing in unfiltered list";  exit 1; }
echo "$out" | grep -q gamma || { echo "FAIL: gamma missing in unfiltered list"; exit 1; }

# 3b. --status pending → alpha + gamma only.
out=$(py list --status pending)
echo "$out" | grep -q alpha || { echo "FAIL: alpha missing under --status pending"; exit 1; }
echo "$out" | grep -q gamma || { echo "FAIL: gamma missing under --status pending"; exit 1; }
if echo "$out" | grep -q beta; then
    echo "FAIL: beta (done) appeared under --status pending"
    exit 1
fi

# 3c. --status done → beta only.
out=$(py list --status done)
echo "$out" | grep -q beta || { echo "FAIL: beta missing under --status done"; exit 1; }
if echo "$out" | grep -q alpha; then
    echo "FAIL: alpha (pending) appeared under --status done"
    exit 1
fi
if echo "$out" | grep -q gamma; then
    echo "FAIL: gamma (pending) appeared under --status done"
    exit 1
fi

echo OK
