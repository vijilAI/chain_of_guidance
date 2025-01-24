from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Metric(ABC):
    """
    Base class for all metrics.

    :param AutoTokenizer tokenizer: The tokenizer to use for tokenizing the input text.
    :param AutoModelForSequenceClassification model: The model to use for scoring the input text.
    """
    def __init__(self, tokenizer=None, model=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)

    @abstractmethod
    def score(self, *args, **kwargs):
        pass
