from utils import preprocess_classes
from .base import BaseEvaluator


class F1Evaluator(BaseEvaluator):
    def evaluate(self, ground_truth: str, prediction: str) -> float:
        gt = preprocess_classes(ground_truth)
        pred = preprocess_classes(prediction)
        if not gt or not pred:
            return 0.0
        correct = gt & pred
        precision = len(correct) / len(pred)
        recall = len(correct) / len(gt)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @property
    def name(self) -> str:
        return "F1"
