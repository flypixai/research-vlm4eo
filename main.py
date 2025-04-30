from argparse import ArgumentParser
from pathlib import Path
from config import Config

from assemble import process_results_directory
from evaluation import evaluate
from inference import infer


def main():
    parser = ArgumentParser()
    parser.add_argument("task", choices=["inference", "assemble", "evaluation"])
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))

    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    match args.task:
        case "inference":
            infer(config)
        case "assemble":
            process_results_directory(config)
        case "evaluation":
            evaluate(config)
        case _:
            raise ValueError(f"Unrecognized task {args.task}")


if __name__ == "__main__":
    main()
