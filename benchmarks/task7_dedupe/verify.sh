#!/bin/sh
# Verify task7_dedupe:
#   1. The four public functions still return the same values and still raise on negatives.
#   2. The duplicated guard line and the rounding call have been deduplicated
#      (each must appear ≤ 1 time in tax.py).
set -e

# Behaviour check.
python - <<'PY'
import importlib.util, math
spec = importlib.util.spec_from_file_location("tax", "tax.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

cases = {
    "calc_with_tax_us": (100, 107.5),
    "calc_with_tax_eu": (100, 120.0),
    "calc_with_tax_jp": (100, 110.0),
    "calc_with_tax_uk": (100, 120.0),
}
for fn_name, (inp, expected) in cases.items():
    fn = getattr(m, fn_name, None)
    assert fn is not None, f"{fn_name} was removed"
    got = fn(inp)
    assert math.isclose(got, expected, abs_tol=0.01), f"{fn_name}({inp}) = {got}, expected {expected}"
    try:
        fn(-1)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{fn_name}(-1) did not raise ValueError")
print("behaviour OK")
PY

# Deduplication check: the guard message and the round call should appear at most once.
guard=$(grep -c 'price must be non-negative' tax.py)
if [ "$guard" -gt 1 ]; then
    echo "FAIL: duplicated guard 'price must be non-negative' appears $guard times"
    exit 1
fi

rounds=$(grep -cE 'round\(.*price.*' tax.py)
if [ "$rounds" -gt 1 ]; then
    echo "FAIL: duplicated rounding logic appears $rounds times"
    exit 1
fi

echo OK
