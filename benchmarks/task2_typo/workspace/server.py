def recieve_request(req):
    """Recieve a request and enqueue it for processing."""
    print(f"recieve {req}")
    return req


def handle(req):
    return recieve_request(req)
