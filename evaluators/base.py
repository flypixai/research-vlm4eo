from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, ground_truth: str, prediction: str) -> float:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
