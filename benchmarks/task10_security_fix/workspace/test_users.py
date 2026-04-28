"""Existing tests for normal find_user behaviour. Must continue to pass."""
import os
import subprocess

import pytest

from users import find_user, list_users


@pytest.fixture(autouse=True)
def fresh_db():
    """Re-create the SQLite database before every test."""
    subprocess.run(["python", "init_db.py"], check=True, capture_output=True)
    yield
    if os.path.exists("users.db"):
        os.remove("users.db")


def test_find_alice():
    rows = find_user("alice")
    assert len(rows) == 1
    assert rows[0][1] == "alice"
    assert rows[0][2] == "admin"


def test_find_bob():
    rows = find_user("bob")
    assert len(rows) == 1
    assert rows[0][1] == "bob"


def test_find_unknown_returns_empty():
    rows = find_user("nobody")
    assert rows == []


def test_list_users_returns_all():
    rows = list_users()
    assert len(rows) == 3
    names = sorted(r[1] for r in rows)
    assert names == ["alice", "bob", "carol"]
