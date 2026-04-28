#!/bin/sh
# Verify task13_typescript:
#   1. package.json and tsconfig.json must be untouched.
#   2. src/calc.ts must export a `subtract` function.
#   3. tsc passes (type check).
#   4. node:test runner passes (existing add/multiply + new subtract).
#   5. The behaviour `subtract(5, 3) === 2` and `subtract(0, 4) === -4` holds.
set -e

# 1. Anti-tampering on config files.
expected_pkg_sha="91ea43af35ab29dba06344c7b8f97c2a45670c2d59147b35f63c417b3250491a"
expected_tsc_sha="acbdbd6fe382e54eb96b3bc13ed297fff2717d42e6c7ccfa48e38bd5f7a17f35"
actual_pkg_sha=$(sha256sum package.json | awk '{print $1}')
actual_tsc_sha=$(sha256sum tsconfig.json | awk '{print $1}')
if [ "$actual_pkg_sha" != "$expected_pkg_sha" ]; then
    echo "FAIL: package.json was modified."
    exit 1
fi
if [ "$actual_tsc_sha" != "$expected_tsc_sha" ]; then
    echo "FAIL: tsconfig.json was modified."
    exit 1
fi

# 2. subtract must be defined and exported.
if ! grep -qE 'export\s+function\s+subtract\s*\(' src/calc.ts; then
    echo "FAIL: src/calc.ts does not export a subtract function"
    exit 1
fi

# 3. node_modules is pre-populated in the workspace, so no install is needed.
# Sanity-check it's actually there (catches the case of a model deleting it).
if [ ! -d node_modules ]; then
    echo "FAIL: node_modules is missing — workspace was not copied correctly"
    exit 1
fi

# 4. Type check.
npx --no-install tsc --noEmit

# 5. Run tests (existing + new).
test_output=$(npx --no-install tsx --test src/calc.test.ts 2>&1) || {
    echo "FAIL: node:test failed"
    echo "$test_output"
    exit 1
}
echo "$test_output" | tail -5

# 6. Behaviour spot-check via tsx eval.
result=$(npx --no-install tsx -e "
import('./src/calc.ts').then(m => {
    if (m.subtract(5, 3) !== 2) { console.error('FAIL: subtract(5,3) =', m.subtract(5,3)); process.exit(1); }
    if (m.subtract(0, 4) !== -4) { console.error('FAIL: subtract(0,4) =', m.subtract(0,4)); process.exit(1); }
    console.log('subtract OK');
})
") || {
    echo "$result"
    exit 1
}
echo "$result"

echo OK
