"""Subcommand handlers for the tasks CLI."""
from store import load, save


def add(args):
    tasks = load(args.store)
    next_id = max((t["id"] for t in tasks), default=0) + 1
    tasks.append({"id": next_id, "title": args.title, "done": False})
    save(tasks, args.store)
    print(f"Added task {next_id}: {args.title}")


def list_tasks(args):
    tasks = load(args.store)
    for t in tasks:
        mark = "x" if t["done"] else " "
        print(f"[{mark}] {t['id']}: {t['title']}")


def done(args):
    tasks = load(args.store)
    for t in tasks:
        if t["id"] == args.id:
            t["done"] = True
            save(tasks, args.store)
            print(f"Marked task {args.id} as done")
            return
    print(f"No task {args.id}")


def remove(args):
    tasks = load(args.store)
    new_tasks = [t for t in tasks if t["id"] != args.id]
    if len(new_tasks) == len(tasks):
        print(f"No task {args.id}")
        return
    save(new_tasks, args.store)
    print(f"Removed task {args.id}")
