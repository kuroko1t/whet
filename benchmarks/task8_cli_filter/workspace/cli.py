"""Tasks CLI entry point."""
import argparse

from handlers import add, list_tasks, done, remove


def build_parser():
    parser = argparse.ArgumentParser(prog="tasks")
    parser.add_argument(
        "--store",
        default=None,
        help="path to the tasks JSON file (default: ~/.tasks.json)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="add a task")
    p_add.add_argument("title")
    p_add.set_defaults(func=add)

    p_list = sub.add_parser("list", help="list tasks")
    p_list.set_defaults(func=list_tasks)

    p_done = sub.add_parser("done", help="mark a task as done")
    p_done.add_argument("id", type=int)
    p_done.set_defaults(func=done)

    p_remove = sub.add_parser("remove", help="remove a task")
    p_remove.add_argument("id", type=int)
    p_remove.set_defaults(func=remove)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
