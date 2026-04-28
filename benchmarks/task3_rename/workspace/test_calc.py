from calc import compute


def test_compute():
    assert compute(2, 3) == 5


if __name__ == "__main__":
    test_compute()
    print("OK")
