from utils import format_output, parse_input


def process_file(filepath):
    """Read a CSV file, parse each line, and print formatted output."""
    results = []
    with open(filepath) as f:
        for line in f:
            record = parse_input(line)
            if record:
                results.append(format_output(record))
    return results


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python app.py <input.csv>")
        sys.exit(1)
    for line in process_file(sys.argv[1]):
        print(line)


if __name__ == "__main__":
    main()
