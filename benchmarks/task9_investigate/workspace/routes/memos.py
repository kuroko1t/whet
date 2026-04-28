"""Memo CRUD routes."""

routes = [
    ("GET",    "/memos",       "list_memos"),
    ("POST",   "/memos",       "create_memo"),
    ("GET",    "/memos/<id>",  "get_memo"),
    ("DELETE", "/memos/<id>",  "delete_memo"),
]


def list_memos():
    return {"memos": []}


def create_memo(payload):
    return {"created": payload}


def get_memo(mid):
    return {"id": mid}


def delete_memo(mid):
    return {"deleted": mid}
