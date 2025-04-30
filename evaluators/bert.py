import bert_score

from .base import BaseEvaluator


class BERTScoreEvaluator(BaseEvaluator):
    def __init__(
        self, model_name: str = "microsoft/deberta-xlarge-mnli", lang: str = "en"
    ):
        self.model_name = model_name
        self.lang = lang

    def evaluate(self, ground_truth: str, prediction: str) -> float:
        _, _, F1 = bert_score.score(
            [prediction],
            [ground_truth],
            model_type=self.model_name,
            lang=self.lang,
            verbose=False,
        )
        return F1.item()

    @property
    def name(self) -> str:
        return "BERTScore"
