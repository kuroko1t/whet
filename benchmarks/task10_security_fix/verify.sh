#!/bin/sh
# Verify task10_security_fix:
#   1. test_users.py and init_db.py must be untouched (anti-tampering).
#   2. Existing tests must still pass.
#   3. The classic SQL injection payload must no longer return all rows.
#   4. users.py must use a parameterized query (no f-string / format-string
#      interpolation of `name` into SQL).
set -e

# 1. Anti-tampering on tests + DB init.
expected_test="bb869fe04ccf2615144fddb56c6a54df282f87109e1018257de138b3c4b4e5d5"
expected_init="0f69791c70d1cab76b4e772fb09398993117a05fbebc7abed637476202f3ce81"
actual_test=$(sha256sum test_users.py | awk '{print $1}')
actual_init=$(sha256sum init_db.py | awk '{print $1}')
if [ "$actual_test" != "$expected_test" ]; then
    echo "FAIL: test_users.py was modified."
    echo "  expected: $expected_test"
    echo "  actual:   $actual_test"
    exit 1
fi
if [ "$actual_init" != "$expected_init" ]; then
    echo "FAIL: init_db.py was modified."
    exit 1
fi

# 2. Existing tests still pass.
python init_db.py >/dev/null
python -m pytest -q test_users.py

# 3. SQL-injection payload must return zero rows.
python init_db.py >/dev/null
python - <<'PY'
import sys
from users import find_user
payload = "' OR '1'='1"
rows = find_user(payload)
if len(rows) != 0:
    print(f"FAIL: SQL injection still works — find_user({payload!r}) returned {len(rows)} rows")
    sys.exit(1)
print("SQL injection blocked")
PY

# 4. Static check: no string interpolation of name into SQL in users.py.
if grep -nE "execute\([^,)]*(f\"|f')" users.py; then
    echo "FAIL: users.py still uses an f-string inside execute(...)"
    exit 1
fi
if grep -nE "execute\([^,)]*\.format\(" users.py; then
    echo "FAIL: users.py still uses .format() inside execute(...)"
    exit 1
fi
if grep -nE "execute\([^,)]*%[^,)]*\)$" users.py; then
    echo "FAIL: users.py still uses % formatting inside execute(...)"
    exit 1
fi

echo OK
