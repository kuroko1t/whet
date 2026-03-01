import sys
from buggy import calculate_average, calculate_median, calculate_std_dev

errors = 0

# Test 1: Normal case (should pass)
assert calculate_average([10, 20, 30]) == 20.0

# Test 2: Empty list average
try:
    result = calculate_average([])
    assert result == 0, f"Expected 0, got {result}"
except Exception as e:
    print(f"FAIL: calculate_average([]) raised {e}")
    errors += 1

# Test 3: Empty list median
try:
    result = calculate_median([])
    assert result == 0, f"Expected 0, got {result}"
except Exception as e:
    print(f"FAIL: calculate_median([]) raised {e}")
    errors += 1

# Test 4: Empty list std_dev
try:
    result = calculate_std_dev([])
    assert result == 0, f"Expected 0, got {result}"
except Exception as e:
    print(f"FAIL: calculate_std_dev([]) raised {e}")
    errors += 1

if errors:
    print(f"\n{errors} TESTS FAILED")
    sys.exit(1)
print("ALL TESTS PASSED")
