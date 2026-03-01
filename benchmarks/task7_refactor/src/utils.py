def format_output(record):
    """Format a record dict into a human-readable string."""
    name = record.get("name", "Unknown")
    score = record.get("score", 0)
    status = "PASS" if score >= 60 else "FAIL"
    return f"{name}: {score} ({status})"


def parse_input(raw):
    """Parse a raw input line into a record dict."""
    parts = raw.strip().split(",")
    if len(parts) != 2:
        return None
    return {"name": parts[0].strip(), "score": int(parts[1].strip())}
