from nltk.translate.meteor_score import single_meteor_score

from .base import BaseEvaluator


# make sure you have these downloaded:
# nltk.download('wordnet')
# nltk.download('omw-1.4')


class MeteorEvaluator(BaseEvaluator):
    def evaluate(self, ground_truth: str, prediction: str) -> float:
        return single_meteor_score(ground_truth, prediction)

    @property
    def name(self) -> str:
        return "METEOR"
