"""Periodic cleanup worker. Doesn't go through process_event — runs
synchronously inside the cron loop."""


def sweep_stale_sessions():
    return 0


def sweep_orphaned_uploads():
    return 0
