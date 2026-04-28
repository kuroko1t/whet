"""Entry point for the simple data pipeline."""
from worker import process_batch
from utils import load_config


def main():
    print("starting pipeline")
    cfg = load_config()
    print(f"loaded config: {cfg}")
    result = process_batch([1, 2, 3, 4, 5])
    print(f"pipeline finished, processed {len(result)} items")
    return result


if __name__ == "__main__":
    main()
