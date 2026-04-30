from eventbus import process_event


def send_email(user_id, subject, body):
    event = {"kind": "email", "to": user_id, "subject": subject, "body": body}
    return process_event(event)


def send_sms(user_id, message):
    return {"sms": user_id, "msg": message}


def fanout_to_subscribers(channel, payload):
    for subscriber_id in _subscribers_of(channel):
        event = {"kind": "channel_msg", "to": subscriber_id, "payload": payload}
        process_event(event)


def _subscribers_of(channel):
    return []
