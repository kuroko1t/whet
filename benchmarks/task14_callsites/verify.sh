#!/bin/sh
# Verify task14_callsites:
#   1. Source files must be untouched (read-only investigation task).
#   2. CALLSITES.md must list exactly the 9 real call sites of `process_event`.
set -e

# 1. Anti-tampering. Pin source SHA-256 so the model can't "fix" the answer
# by editing the codebase to delete callsites or moving the definition.
expected_eventbus="6d52d0e602959c37516ca5269667bf17de4cb851d447fcddd41213fa9161f66e"
expected_login="7e1c63685ef6ed8a465ce97bc6e8f8cec2592cc0a43012247aeef51fdf1894bd"
expected_payment="ae07ac9ca3550152c8be908fd46b8f4a35bad7d782ce02adbd494077e4da4710"
expected_notification="fe4706620cb41d7c867d92071e78c876c43e3d1bceb105cd726610c13228386c"
expected_admin="0088e493e4e5bc1548181d7bc30cee86580dd543853043c8faae94d69ce8fe43"
expected_email_w="88e7c1d892f1863a356b87344e57a31705df68dc1607687b627989966a49d1c6"
expected_billing_w="e18f9588766d589f3a0ebca1b9483778ccc2adaecdc8a119c230b22930ee30c8"
expected_cleanup_w="0e30fcb8ebe9e1e2a7658c99061bc858d3e78c5c3032da9995296d7d30e763f1"
expected_retry_w="161ae7be67e519641d8152ea5c3f6cc60375c5eb87bc7cc332e6699b67d26392"
expected_metrics="02e54733595e6a13f1473b857e71212d2687a237974e9e9f1b5ae8f3712e05b3"
expected_cli="4a1a4238a7d16e89364f397a40c4e0107b6a2560fca06c48087ef30051d109be"

check_sha() {
    local file="$1" expected="$2"
    local actual
    actual=$(sha256sum "$file" | awk '{print $1}')
    if [ "$actual" != "$expected" ]; then
        echo "FAIL: source file '$file' was modified by the agent (read-only task)."
        echo "  expected: $expected"
        echo "  actual:   $actual"
        exit 1
    fi
}
check_sha "eventbus.py" "$expected_eventbus"
check_sha "handlers/login.py" "$expected_login"
check_sha "handlers/payment.py" "$expected_payment"
check_sha "handlers/notification.py" "$expected_notification"
check_sha "handlers/admin.py" "$expected_admin"
check_sha "workers/email_worker.py" "$expected_email_w"
check_sha "workers/billing_worker.py" "$expected_billing_w"
check_sha "workers/cleanup_worker.py" "$expected_cleanup_w"
check_sha "workers/retry_worker.py" "$expected_retry_w"
check_sha "utils/metrics.py" "$expected_metrics"
check_sha "cli.py" "$expected_cli"

# 2. CALLSITES.md must exist.
if [ ! -f CALLSITES.md ]; then
    echo "FAIL: CALLSITES.md was not created"
    exit 1
fi

# 3. The set of expected (path, line) pairs must be a subset of what's listed.
# We tolerate extra correct entries (the model is allowed to be over-thorough)
# but reject any false-positive (a path:line that does NOT actually call
# process_event in the source).
python3 - <<'PY'
import re
import sys
import pathlib

expected = {
    ("handlers/login.py", 6),
    ("handlers/notification.py", 6),
    ("handlers/notification.py", 16),
    ("handlers/payment.py", 6),
    ("handlers/payment.py", 12),
    ("workers/billing_worker.py", 16),
    ("workers/email_worker.py", 6),
    ("workers/email_worker.py", 13),
    ("workers/retry_worker.py", 6),
}

answer = pathlib.Path("CALLSITES.md").read_text()

# Parse `path:line` entries — accept optional leading bullets/numbers, accept
# trailing `:column` if the model emitted grep-style output.
got = set()
for raw_line in answer.splitlines():
    line = raw_line.strip().lstrip("-*`#0123456789. ").strip("`*")
    # Tolerate `./path/to/file.py:6` and `path:line:col`.
    m = re.match(r"^\.?/?([A-Za-z0-9_./-]+\.py):(\d+)(?::\d+)?", line)
    if not m:
        continue
    path = m.group(1)
    # Strip leading "./" if present
    if path.startswith("./"):
        path = path[2:]
    got.add((path, int(m.group(2))))

missing = expected - got
extra = got - expected

# Cross-check extras: every reported entry that's NOT in expected must
# correspond to an actual `process_event(` call in the source. If not, it's
# a false positive (e.g. the model listed a docstring mention).
false_positives = []
for path, line in extra:
    p = pathlib.Path(path)
    if not p.exists():
        false_positives.append((path, line, "file does not exist"))
        continue
    lines = p.read_text().splitlines()
    if line < 1 or line > len(lines):
        false_positives.append((path, line, "line out of range"))
        continue
    src_line = lines[line - 1]
    # Strip everything after `#` (comment) and find a bare `process_event(`
    # that's not inside a string literal. Quick-and-dirty: kill everything
    # between matching quotes first, then check.
    code = re.sub(r'#.*$', '', src_line)
    code = re.sub(r'"[^"]*"', '""', code)
    code = re.sub(r"'[^']*'", "''", code)
    if "process_event(" not in code:
        false_positives.append((path, line, f"no call here: {src_line.strip()!r}"))

if missing or false_positives:
    if missing:
        print("FAIL: missing call sites:")
        for p, l in sorted(missing):
            print(f"  - {p}:{l}")
    if false_positives:
        print("FAIL: false positives reported:")
        for p, l, reason in sorted(false_positives):
            print(f"  - {p}:{l} ({reason})")
    print("---- CALLSITES.md ----")
    print(answer)
    print("---- parsed ----")
    for p, l in sorted(got):
        print(f"  {p}:{l}")
    sys.exit(1)

print(f"OK ({len(expected)} call sites, {len(extra)} valid extras)")
PY
