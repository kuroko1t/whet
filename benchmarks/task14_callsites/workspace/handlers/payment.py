from eventbus import process_event


def on_charge(amount, user_id):
    event = {"kind": "charge", "amount": amount, "user": user_id}
    result = process_event(event)
    return result


def on_refund(charge_id):
    event = {"kind": "refund", "charge": charge_id}
    return process_event(event)


def on_dispute(charge_id):
    log = "user disputed; would call process_event but we don't on disputes"
    return {"disputed": charge_id, "log": log}
