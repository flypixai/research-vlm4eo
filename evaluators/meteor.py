import nltk
from nltk.translate.meteor_score import single_meteor_score

from .base import BaseEvaluator


class MeteorEvaluator(BaseEvaluator):
    def __init__(self):
        # must have these downloaded:
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    def evaluate(self, ground_truth: str, prediction: str) -> float:
        return single_meteor_score(ground_truth.split(), prediction.split())

    @property
    def name(self) -> str:
        return "METEOR"
