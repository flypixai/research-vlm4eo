import plotly.express as px
import polars as pl
from tqdm import tqdm

from config import ASSEMBLED_FILENAME, Config, Models, SEPARATOR, get_task_dirname

EVALUATION_CSV = "evaluation.csv"


def build_results(config: Config):
    for i, task in enumerate(config.tasks):
        task_path = config.output_folder / get_task_dirname(i)
        save_path = task_path / EVALUATION_CSV

        if save_path.is_file():
            print(f"{save_path.as_posix()} exists, skipping...")
            continue

        df = pl.read_csv(task_path / ASSEMBLED_FILENAME, separator=SEPARATOR)

        models = [m.value for m in Models.all()]
        humans = [mp for mp in df["model/person"] if mp not in models]

        schema = {"image": str, "human": str, "metric": str}
        schema.update({m.value: float for m in config.models})

        evaluators = [e.get() for e in config.evaluation_map[i]]

        images = df.columns
        images.remove("model/person")

        rows = []
        for image in tqdm(images):
            for evaluator in evaluators:
                for human in humans:
                    row = {"image": image, "human": human, "metric": evaluator.name}
                    for actor in models + humans:
                        if actor == human:
                            continue
                        human_output = df.filter(pl.col("model/person") == human)[
                            image
                        ][0]
                        model_output = df.filter(pl.col("model/person") == actor)[
                            image
                        ][0]
                        result = evaluator.evaluate(human_output, model_output)
                        row.update({actor: result})
                    rows.append(row)

        df = pl.DataFrame(rows)
        df.write_csv(save_path)


def plot(config: Config):
    # TODO: Some things I quickly hardcoded here, could be improved
    for i, task in enumerate(config.tasks):
        task_path = config.output_folder / get_task_dirname(i)
        csv_path = task_path / EVALUATION_CSV
        df = pl.read_csv(csv_path)

        df_long = df.melt(
            id_vars=["image", "human", "metric"],
            variable_name="model",
            value_name="score",
        )

        include_models = [m.value for m in Models.all()]
        included = df_long.filter(pl.col("model").is_in(include_models))
        excluded = df_long.filter(~pl.col("model").is_in(include_models))

        included_avg = included.group_by(["metric", "model"]).agg(
            pl.col("score").mean().alias("score")
        )
        included_avg = included_avg.sort(by="model")

        fig = px.histogram(
            included_avg,
            x="model",
            y="score",
            color="metric",
            barmode="group",
            histfunc="avg",
            height=400,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            category_orders={
                "metric": ["METEOR", "BERTScore", "Precision", "Recall", "F1"]
            },
            labels={"score": "score"},
        )

        fig.update_layout(yaxis_title="score")

        color_map = {"METEOR": 0, "BERTScore": 1, "Precision": 2, "Recall": 3, "F1": 4}

        excluded_avg = excluded.group_by(["metric"]).agg(
            pl.col("score").mean().alias("score")
        )
        for i, row in enumerate(excluded_avg.iter_rows()):
            if row[0] in ["Precision", "Recall"]:
                continue
            text = "human" if row[0] != "F1" else "human F1"
            fig.add_hline(
                y=row[1],
                line_dash=["dash", "dot", "longdash"][i],
                line_color=px.colors.qualitative.Pastel[color_map[row[0]]],
                annotation_text=text,
                annotation_position="top right",
            )

        fig.write_image(task_path / "plot.png")


def evaluate(config: Config):
    build_results(config)
    plot(config)
