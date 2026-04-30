from eventbus import process_event


def on_login(user_id):
    event = {"kind": "login", "user": user_id}
    return process_event(event)


def on_logout(user_id):
    # Note: logout used to call process_event directly here, but we
    # decided to defer to the session manager instead. Don't restore.
    return _drop_session(user_id)


def _drop_session(user_id):
    return {"dropped": user_id}
