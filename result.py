import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel


class IOPair(BaseModel):
    task: str
    output: str


class Result(BaseModel):
    model: str
    image_name: str
    io_pairs: list[IOPair]

    def save_as_json(self, save_path: Path):
        with open(save_path, "w") as jf:
            json.dump(self.model_dump(), jf)

    @classmethod
    def from_json(cls, json_path: Path) -> Self:
        with open(json_path, "r") as jf:
            data = json.load(jf)
        return cls(**data)
