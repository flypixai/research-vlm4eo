import polars as pl

from config import Config, HUMAN_FILENAME, RESULTS_FILENAME, get_task_dirname


def evaluate(config: Config):
    for i, task in config.tasks:
        task_path = config.output_folder / get_task_dirname(i)
        model_df = pl.read_csv(task_path / RESULTS_FILENAME, separator=";")

        # TODO:
        human_df = pl.read_csv(task_path / HUMAN_FILENAME, separator=";")
        humans = ...  # list of strings "human_0", "human_1", ...

        schema = {"image": str, "human": str, "metric": str}
        schema.update({m.value: float for m in config.models})

        evaluators = [e.get() for e in config.evaluation_map[i]]

        rows = []
        for image in model_df.image:  # have to fetch the image names from here, check
            for evaluator in evaluators:
                evaluator = evaluator.get()
                for human in humans:
                    row = {"image": image, "human": human, "metric": evaluator.name}
                    for model in config.models:
                        human_output = ...  # have to get it here
                        model_output = ...  # have to get it here
                        result = evaluator.evaluate(human_output, model_output)
                        row.update({model.value: result})
                    rows.append(row)

        df = pl.DataFrame(rows)
        df.write_csv(config.output_folder / f"evaluation_task_{i}.csv")
