from enum import Enum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field

from evaluators.base import BaseEvaluator
from evaluators.meteor import MeteorEvaluator
from evaluators.precision import PrecisionEvaluator
from vllms import Gemma3, JanusPro7B, Phi4
from vllms.base import BaseVLLM

RESULTS_FILENAME = "results.csv"
HUMAN_FILENAME = "human.csv"


def get_task_dirname(i: int) -> str:
    return f"task_{i}"


class Models(Enum):
    Gemma3 = "Gemma3"
    JanusPro7B = "JanusPro7B"
    Phi4 = "Phi4"

    @classmethod
    def all(cls) -> list[Self]:
        return [cls.Gemma3, cls.JanusPro7B, cls.Phi4]

    def get(self) -> BaseVLLM:
        match self:
            case self.Gemma3:
                return Gemma3()
            case self.JanusPro7B:
                return JanusPro7B()
            case self.Phi4:
                return Phi4()


class Evaluators(Enum):
    Meteor = "Meteor"
    Precision = "Precision"
    Recall = "Recall"

    def get(self) -> BaseEvaluator:
        match self:
            case self.Meteor:
                return MeteorEvaluator()
            case self.Precision:
                return PrecisionEvaluator()


class Config(BaseModel):
    input_folder: Path = Path("images/")
    output_folder: Path = Path("results")
    human_outputs: Path = Path("human/")
    models: list[Models] = Field(default_factory=lambda: Models.all)
    tasks: list[str] = [
        "Describe the image",
        "List all object classes (one-two words). Use comma to separate the classes. Only output. No extra text.",
        "List all landzones (one-two words). Use comma to separate the landzones. Only output. No extra text.",
    ]
    evaluation_map: dict[int, list[Evaluators]] = Field(
        default_factory=lambda: {
            0: ["Meteor"],
            1: ["Precision"],
            2: ["Precision"],
        }
    )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> Self:
        with open(yaml_path, "r") as yf:
            data = yaml.safe_load(yf)
        return cls(**data)
