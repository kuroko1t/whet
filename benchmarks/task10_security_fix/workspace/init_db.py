"""Recreate users.db with three sample rows."""
import os
import sqlite3

DB = "users.db"
if os.path.exists(DB):
    os.remove(DB)
conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, role TEXT)")
cur.executemany(
    "INSERT INTO users (name, role) VALUES (?, ?)",
    [("alice", "admin"), ("bob", "user"), ("carol", "user")],
)
conn.commit()
conn.close()
print("users.db initialised with 3 rows")
