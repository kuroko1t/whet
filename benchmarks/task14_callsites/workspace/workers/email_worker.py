from eventbus import process_event


def consume_one(job):
    event = {"kind": "email_send_attempt", "job_id": job.get("id")}
    return process_event(event)


def consume_batch(jobs):
    out = []
    for job in jobs:
        event = {"kind": "email_send_attempt", "job_id": job.get("id")}
        out.append(process_event(event))
    return out
