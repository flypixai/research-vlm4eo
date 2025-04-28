from tqdm import tqdm

from config import Config, get_task_dirname


def infer(config: Config):
    input_images = sorted(list(config.input_folder.glob("*")))

    config.output_folder.mkdir(exist_ok=True)

    for i, task in enumerate(config.tasks):
        print(f"Running task: \n {task}")

        task_path = config.output_folder / get_task_dirname(i)
        task_path.mkdir(exist_ok=True)

        with open(task_path / "task.txt", "w") as f:
            f.write(task)

        for model in config.models:
            model = model.get()
            print(f" --- Running model: {model.__class__.__name__}")

            save_path = task_path / model.__class__.__name__
            save_path.mkdir(exist_ok=True)

            for image_path in tqdm(input_images, desc="Images"):
                full_save_path = save_path / f"{image_path.name}.txt"
                if full_save_path.exists():
                    print(
                        f"Output '{full_save_path.as_posix()}' already exists, skipping..."
                    )
                    continue

                out = model.prompt(task, image_path=image_path)
                full_save_path.write_text(out)
