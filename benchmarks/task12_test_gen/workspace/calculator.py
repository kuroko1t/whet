"""A small calculator with a few non-trivial edge cases."""


class Calculator:
    """Stateless calculator that operates on numeric inputs.

    Edge cases worth exercising in a test suite:
      - division by zero (raises ZeroDivisionError)
      - non-numeric input (raises TypeError)
      - negative numbers
      - the `power` method with zero exponent
      - the `average` method on an empty iterable (raises ValueError)
    """

    def add(self, a, b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("inputs must be numeric")
        return a + b

    def subtract(self, a, b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("inputs must be numeric")
        return a - b

    def divide(self, a, b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("inputs must be numeric")
        if b == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return a / b

    def power(self, base, exponent):
        if not isinstance(base, (int, float)) or not isinstance(exponent, (int, float)):
            raise TypeError("inputs must be numeric")
        return base ** exponent

    def average(self, values):
        # Materialize so we can validate type and emptiness.
        values = list(values)
        if not values:
            raise ValueError("cannot take the average of an empty iterable")
        for v in values:
            if not isinstance(v, (int, float)):
                raise TypeError("all values must be numeric")
        return sum(values) / len(values)
