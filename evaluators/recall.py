from utils import preprocess_classes
from .base import BaseEvaluator


class RecallEvaluator(BaseEvaluator):
    def evaluate(self, ground_truth: str, prediction: str) -> float:
        gt = preprocess_classes(ground_truth)
        pred = preprocess_classes(prediction)
        if not gt:
            return 0.0
        correct = gt & pred
        return len(correct) / len(gt)

    @property
    def name(self) -> str:
        return "Recall"
