"""Admin tools. None of these dispatch through the event bus — admin
operations are intentionally synchronous and bypass `process_event`."""


def ban_user(user_id):
    return {"banned": user_id}


def promote_user(user_id):
    return {"promoted": user_id}


def list_audit():
    return []
