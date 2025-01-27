import spacy
import evaluate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .prompt_template import EVAL_STEP1_TEMPLATE, EVAL_STEP2_TEMPLATE

class Agreement:

    def __init__(self):
        pass

class BLEU(Agreement):
    name = "bleu"

    def __init__(self):
        self.bleu = evaluate.load("bleu")

    def score(self, output_i, output_j):
        if not output_i:
            return 0
        if not output_j:
            return 0
        bleu_score = self.bleu.compute(predictions=[output_i], references=[output_j])
        return bleu_score["bleu"] or 0.0
    
class BERTScore(Agreement):
    name = "bertscore"

    def __init__(self):
        self.bertscore = evaluate.load("bertscore")

    def score(self, output_i, output_j):
        bertscore_score = self.bertscore.compute(
            predictions=[output_i], references=[output_j], lang="en"
        )
        return bertscore_score["f1"][0]