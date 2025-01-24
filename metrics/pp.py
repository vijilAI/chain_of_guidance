import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ParaphraseDetector:
    """
    This class implements a Paraphrase Detector using a finetuned LLM. Currently, it uses DeBERTa V3 finetuned on the PAWS dataset.
    It serves to assess if two given text segments are paraphrases of each other.
    The model outputs a score between 0 and 1, indicating the likelihood of the inputs being paraphrases.

    :param AutoTokenizer detection_tokenizer: Hugging Face tokenizer, initialized from a pretrained model specified by `tok_path`.
    :param AutoModelForSequenceClassification detection_model: Hugging Face model fine-tuned for paraphrase detection, loaded from `model_path`.
    :param str tok_path: Path or identifier for the tokenizer to be used. Default is set to "domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector".
    :param str model_path: Path or identifier for the model to be used. Mirrors the default of `tok_path`.

    Example:
        ```
        detector = ParaphraseDetector()
        score_para = detector.score("This is a sentence.", "This is another sentence.")
        ```
        `score_para` represents the probability of them being paraphrases.
    """
    name = "pp"

    def __init__(
        self,
        tok_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector",
        model_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector",
    ):
        self.detection_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        self.cuda_flag = torch.cuda.is_available()
        if self.cuda_flag:
            self.detection_model.to("cuda")

    def score(self, sentence1, sentence2):
        """
        Calculates the paraphrase probability scores for two input sentences.

        Parameters:
            sentence1 (str): The first sentence.
            sentence2 (str): The second sentence.

        Returns:
            (float): Probabilities that the two sentences are each other's paraphrases.
        """
        inputs = self.detection_tokenizer(
            sentence1, sentence2, return_tensors="pt", padding=True
        )
        if self.cuda_flag:
            inputs.to("cuda")
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        return scores.T[1].item()
