"""Metrics emission. The doc string mentions process_event because we
considered routing metrics through it, but rejected that idea — metrics
go to a separate sink. NO real call to process_event in this file."""


_counter = {}


def incr(name, n=1):
    _counter[name] = _counter.get(name, 0) + n


def reset():
    _counter.clear()
