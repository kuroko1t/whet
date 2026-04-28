"""Tiny user lookup module backed by SQLite."""
import sqlite3

DB_PATH = "users.db"


def find_user(name):
    """Return all users whose name exactly matches `name`."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # NOTE: this query interpolates `name` directly into the SQL string.
    cur.execute(f"SELECT id, name, role FROM users WHERE name = '{name}'")
    rows = cur.fetchall()
    conn.close()
    return rows


def list_users():
    """Return every user."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name, role FROM users")
    rows = cur.fetchall()
    conn.close()
    return rows
