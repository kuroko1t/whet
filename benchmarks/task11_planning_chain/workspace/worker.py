"""Worker that processes a batch of integers."""


def process_batch(items):
    print(f"worker: received batch of {len(items)} items")
    out = []
    for i in items:
        if i < 0:
            print(f"worker: skipping invalid item {i}")
            continue
        out.append(i * 2)
    print(f"worker: produced {len(out)} results")
    return out
