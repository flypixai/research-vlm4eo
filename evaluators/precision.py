from utils import split_classes
from .base import BaseEvaluator


class PrecisionEvaluator(BaseEvaluator):
    def evaluate(self, ground_truth: str, prediction: str) -> float:
        gt = split_classes(ground_truth)
        pred = split_classes(prediction)
        if not pred:
            return 0.0
        correct = gt & pred
        return len(correct) / len(pred)

    @property
    def name(self) -> str:
        return "Precision"
