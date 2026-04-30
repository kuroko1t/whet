"""Timing helpers — pure utilities, no event dispatch."""

import time


def now_ms():
    return int(time.time() * 1000)


def deadline_ms(after_ms):
    return now_ms() + after_ms
