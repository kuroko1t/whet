"""CLI entry point. Spins up handlers and workers; doesn't call
process_event itself."""

import sys


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: cli.py <command>")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
