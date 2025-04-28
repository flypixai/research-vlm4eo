from pathlib import Path
import pandas as pd

from result import Result

RESULTS_DIR = "results/"
SAVE_PATH = "results.csv"


def process_results_directory(results_dir: Path) -> pd.DataFrame:
    entries = []

    model_paths = results_dir.glob("*/")

    for model in model_paths:
        files = model.glob("*.json")

        for file in files:
            result = Result.from_json(file)
            for io_pair in result.io_pairs:
                entries.append(
                    {
                        "file": file.name,
                        "task": io_pair.task,
                        "model": result.model,
                        "output": io_pair.output,
                    }
                )

    df = pd.DataFrame(entries)
    if df.empty:
        return pd.DataFrame()

    pivot_df = df.pivot(
        index=["file", "task"], columns="model", values="output"
    ).reset_index()

    # Rename columns for clarity
    pivot_df.columns.name = None  # Remove the 'model' label from columns
    return pivot_df


def main():
    df = process_results_directory(Path(RESULTS_DIR))
    df.to_csv(SAVE_PATH, sep=";")


if __name__ == "__main__":
    main()
