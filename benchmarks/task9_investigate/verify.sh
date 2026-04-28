#!/bin/sh
# Verify task9_investigate:
#   1. Source files must be untouched (this is a read-only task).
#   2. ANSWER.md must exist and list every required endpoint (case-insensitive).
set -e

# 1. Anti-tampering: pin source SHA-256 so the model can't "fix" the answer by editing.
expected_app="45b68de601b15f9cd0494ee952b4a58497e01790e2f3a9be060157486f34a0d9"
expected_users="2b704bcb612f95aaa89408de3927b781cf024fcb161ffe8b1521caf9e906406e"
expected_memos="1c82ec0eec7b1baa8ef1c95637bd8140a4ea274473fb05109c465d69390c05bb"
expected_health="67bfff3233707dd1c94bde3b2cfa82c7843fb7ac7bfa0aca84fba92df4b63454"

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
check_sha "app.py" "$expected_app"
check_sha "routes/users.py" "$expected_users"
check_sha "routes/memos.py" "$expected_memos"
check_sha "routes/health.py" "$expected_health"

# 2. ANSWER.md must exist.
if [ ! -f ANSWER.md ]; then
    echo "FAIL: ANSWER.md was not created"
    exit 1
fi

# 3. Every required endpoint must appear in ANSWER.md.
# We accept any of these path-variable spellings: <id>, <uid>, :id, {id}, {uid}.
# Match is case-insensitive on METHOD only; path is matched as-is on the prefix.
python3 - <<'PY'
import re, sys, pathlib

answer = pathlib.Path("ANSWER.md").read_text()

# (METHOD, base_path)
required = [
    ("GET",    "/users"),
    ("GET",    "/users/"),    # any /users/<id> variant
    ("POST",   "/users"),
    ("DELETE", "/users/"),    # any /users/<id> variant
    ("PATCH",  "/users/"),    # any /users/<id> variant
    ("GET",    "/memos"),
    ("POST",   "/memos"),
    ("GET",    "/memos/"),
    ("DELETE", "/memos/"),
    ("GET",    "/health"),
]

# Build a set of (method, normalized-path) pairs from ANSWER.md.
# A line like "GET /users/<id>" or "DELETE /memos/{id}" → ("GET", "/users/").
found = set()
for raw_line in answer.splitlines():
    line = raw_line.strip().lstrip("-*0123456789. ").strip("`*")
    m = re.match(r'(?i)^\s*(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/[A-Za-z0-9/<>:{}_-]*)', line)
    if not m:
        continue
    method = m.group(1).upper()
    path = m.group(2)
    # Normalize variable segments to the trailing slash form
    norm = re.sub(r'/<[^>]+>', '/', path)
    norm = re.sub(r'/\{[^}]+\}', '/', norm)
    norm = re.sub(r'/:[A-Za-z_]+', '/', norm)
    found.add((method, norm))

missing = []
for method, base in required:
    # Accept either exact match or with trailing slash interchangeably for the base form.
    candidates = {(method, base), (method, base.rstrip('/'))}
    if not (candidates & found):
        missing.append(f"{method} {base}<id>" if base.endswith('/') and base != "/" else f"{method} {base}")

if missing:
    print("FAIL: missing endpoints:")
    for m in missing:
        print(f"  - {m}")
    print("---- ANSWER.md ----")
    print(answer)
    print("---- parsed ----")
    for f in sorted(found):
        print(f"  {f[0]} {f[1]}")
    sys.exit(1)

print(f"OK ({len(required)} endpoints listed)")
PY
