"""Tests for the tasks CLI."""
import json
from pathlib import Path

import pytest

from cli import main


@pytest.fixture
def store(tmp_path):
    return tmp_path / "tasks.json"


def run(argv, store):
    main(["--store", str(store)] + argv)


def test_add_creates_task(store, capsys):
    run(["add", "buy milk"], store)
    data = json.loads(Path(store).read_text())
    assert len(data) == 1
    assert data[0]["title"] == "buy milk"
    assert data[0]["done"] is False


def test_list_shows_tasks(store, capsys):
    run(["add", "first"], store)
    run(["add", "second"], store)
    capsys.readouterr()
    run(["list"], store)
    out = capsys.readouterr().out
    assert "first" in out
    assert "second" in out


def test_done_marks_task(store):
    run(["add", "task A"], store)
    run(["done", "1"], store)
    data = json.loads(Path(store).read_text())
    assert data[0]["done"] is True


def test_remove_deletes_task(store):
    run(["add", "task A"], store)
    run(["add", "task B"], store)
    run(["remove", "1"], store)
    data = json.loads(Path(store).read_text())
    assert len(data) == 1
    assert data[0]["id"] == 2
