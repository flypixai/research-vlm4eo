from enum import Enum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field

from evaluators.base import BaseEvaluator
from evaluators.bert import BERTScoreEvaluator
from evaluators.f1 import F1Evaluator
from evaluators.meteor import MeteorEvaluator
from evaluators.precision import PrecisionEvaluator
from evaluators.recall import RecallEvaluator
from vllms import Gemma3, JanusPro7B, Phi4, Qwen2_5_7B, SmolVLM
from vllms.base import BaseVLLM

ASSEMBLED_FILENAME = "assembled.csv"
HUMAN_FILENAME = "human.csv"

SEPARATOR = "|"


def get_task_dirname(i: int) -> str:
    return f"task_{i}"


class Models(Enum):
    Gemma3 = "Gemma3"
    JanusPro7B = "JanusPro7B"
    Phi4 = "Phi4"
    Qwen2_5 = "Qwen2_5_7B"
    SmolVLM = "SmolVLM"

    @classmethod
    def all(cls) -> list[Self]:
        return [cls.Gemma3, cls.JanusPro7B, cls.Phi4, cls.Qwen2_5, cls.SmolVLM]

    def get(self) -> BaseVLLM:
        match self:
            case self.Gemma3:
                return Gemma3()
            case self.JanusPro7B:
                return JanusPro7B()
            case self.Phi4:
                return Phi4()
            case self.Qwen2_5:
                return Qwen2_5_7B()
            case self.SmolVLM:
                return SmolVLM()


class Evaluators(Enum):
    Meteor = "Meteor"
    BERTScore = "BERTScore"
    Precision = "Precision"
    Recall = "Recall"
    F1 = "F1"

    def get(self) -> BaseEvaluator:
        match self:
            case self.Meteor:
                return MeteorEvaluator()
            case self.BERTScore:
                return BERTScoreEvaluator()
            case self.Precision:
                return PrecisionEvaluator()
            case self.Recall:
                return RecallEvaluator()
            case self.F1:
                return F1Evaluator()


class Config(BaseModel):
    input_folder: Path = Path("images/")
    output_folder: Path = Path("results/")
    human_outputs: Path = Path("human/")
    models: list[Models] = Field(default_factory=lambda: Models.all)
    tasks: list[str] = [
        "Describe the image",
        "List all object classes (one-two words). Use comma to separate the classes. Only output. No extra text.",
        "List all landzones (one-two words). Use comma to separate the landzones. Only output. No extra text.",
    ]
    evaluation_map: dict[int, list[Evaluators]] = Field(
        default_factory=lambda: {
            0: [Evaluators.Meteor, Evaluators.BERTScore],
            1: [Evaluators.Precision, Evaluators.Recall, Evaluators.F1],
            2: [Evaluators.Precision, Evaluators.Recall, Evaluators.F1],
        }
    )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> Self:
        with open(yaml_path, "r") as yf:
            data = yaml.safe_load(yf)
        return cls(**data)
