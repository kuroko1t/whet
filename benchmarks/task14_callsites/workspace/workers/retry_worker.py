from eventbus import process_event


def retry(job, attempt):
    event = {"kind": "retry", "job": job, "attempt": attempt}
    return process_event(event)
