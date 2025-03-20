import abc
from pathlib import Path


class BaseVLLM(abc.ABC):
    @abc.abstractmethod
    def prompt(self, instruction: str, image_path: Path) -> str:
        ...
