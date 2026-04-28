"""User-related HTTP routes."""

routes = [
    ("GET",    "/users",         "list_users"),
    ("GET",    "/users/<id>",    "get_user"),
    ("POST",   "/users",         "create_user"),
    ("DELETE", "/users/<id>",    "delete_user"),
    ("PATCH",  "/users/<id>",    "update_user"),
]


def list_users():
    return {"users": []}


def get_user(uid):
    return {"id": uid}


def create_user(payload):
    return {"created": payload}


def delete_user(uid):
    return {"deleted": uid}


def update_user(uid, payload):
    return {"updated": uid, "fields": payload}
