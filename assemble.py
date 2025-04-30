import polars as pl

from config import ASSEMBLED_FILENAME, Config, SEPARATOR

RESULTS_DIR = "results/"
HUMAN_FILENAME = "human.csv"


def process_results_directory(config: Config) -> pl.DataFrame:
    task_paths = [d for d in config.output_folder.glob("*") if d.is_dir()]

    print(f"Assembling results for {len(list(task_paths))} tasks.")

    for task_path in task_paths:
        model_dirs = [d for d in task_path.glob("*") if d.is_dir()]

        rows = []
        for model_dir in model_dirs:
            result_paths = list(model_dir.glob("*.txt"))
            result_paths = sorted(result_paths)
            new_row = {"model/person": model_dir.name}
            for result_path in result_paths:
                new_row.update({result_path.stem: result_path.read_text()})
            rows.append(new_row)

        df = pl.DataFrame(rows)

        human_df_path = task_path / HUMAN_FILENAME
        if human_df_path.is_file():
            human_df = pl.read_csv(human_df_path, separator=";")

            human_df = human_df.transpose(include_header=True)
            new_columns = [v[0] for v in human_df[0].to_dict().values()]
            human_df.columns = new_columns
            human_df = human_df.slice(1)
            human_df = human_df.rename({"image": "model/person"})
            human_df = human_df.select(df.columns)
            df = human_df.vstack(df)

        save_path = task_path / ASSEMBLED_FILENAME
        df.write_csv(save_path, separator=SEPARATOR)
        print(f"Wrote results to {save_path.as_posix()}.")


def main():
    process_results_directory(Config())


if __name__ == "__main__":
    main()
