"""Tests that observe the pipeline through `caplog` (the logging-style way).

These tests fail today because the source modules use `print()` instead of
the `logging` module. After migration they must pass without modification.
"""
import logging

import pytest

from main import main
from worker import process_batch
from utils import load_config


def test_main_emits_pipeline_start(caplog):
    with caplog.at_level(logging.INFO):
        main()
    msgs = " ".join(record.getMessage() for record in caplog.records)
    assert "starting pipeline" in msgs, "expected an INFO log when the pipeline starts"


def test_main_logs_finished(caplog):
    with caplog.at_level(logging.INFO):
        main()
    msgs = " ".join(record.getMessage() for record in caplog.records)
    assert "pipeline finished" in msgs


def test_worker_logs_batch_size(caplog):
    with caplog.at_level(logging.INFO):
        process_batch([10, 20, 30])
    msgs = " ".join(record.getMessage() for record in caplog.records)
    assert "batch of 3 items" in msgs


def test_worker_logs_skip_at_warning_or_higher(caplog):
    with caplog.at_level(logging.DEBUG):
        process_batch([1, -1, 2])
    skip_records = [r for r in caplog.records if "skipping invalid item -1" in r.getMessage()]
    assert skip_records, "the 'skipping invalid item' message must appear in the log"
    # It should be a warning or higher (skipping bad input is noteworthy, not just info)
    assert any(r.levelno >= logging.WARNING for r in skip_records), \
        "the skip message should be logged at WARNING or higher"


def test_load_config_logs_at_debug_or_info(caplog):
    with caplog.at_level(logging.DEBUG):
        load_config()
    assert any("loading config" in r.getMessage() for r in caplog.records), \
        "load_config should emit a log line about loading"
