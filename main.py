from pathlib import Path

from tqdm import tqdm

from result import IOPair, Result
from vllms import Gemma3, JanusPro7B, Phi4

INPUT_FOLDER = "images/"
OUTPUT_FOLDER = "results"
MODELS = [Phi4(), JanusPro7B(), Gemma3()]
INSTRUCTIONS = [
    "describe image",
    "list  all objects classes (one-two words) and separatly landzones. use comma",
]

# ---


def main():
    input_images = sorted(list(Path(INPUT_FOLDER).glob("*")))

    output_path = Path(OUTPUT_FOLDER) 
    output_path.mkdir(exist_ok=True)

    for model in MODELS:
        print(f"--- Running {model.__class__.__name__} ---")
        save_path = output_path / model.__class__.__name__
        save_path.mkdir(exist_ok=True)

        for image_path in tqdm(input_images, desc="Images"):
            full_save_path = save_path / f"{image_path.name}.json"
            if full_save_path.exists():
                print(
                    f"Output '{full_save_path.as_posix()}' already exists, skipping..."
                )
                continue

            io_pairs = []
            for instruction in INSTRUCTIONS:
                out = model.prompt(instruction, image_path=image_path)
                io_pairs.append(IOPair(instruction=instruction, output=out))

            result = Result(
                model=model.__class__.__name__,
                image_name=image_path.name,
                io_pairs=io_pairs,
            )
            result.save_as_json(save_path / f"{image_path.name}.json")


if __name__ == "__main__":
    main()
