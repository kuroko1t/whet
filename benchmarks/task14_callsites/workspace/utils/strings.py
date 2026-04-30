"""String helpers — pure utilities."""


def slugify(s):
    return s.lower().replace(" ", "-")


def truncate(s, n):
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"
