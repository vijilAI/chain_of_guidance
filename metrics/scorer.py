import numpy as np
from scipy.stats import entropy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from abc import ABC, abstractmethod

from .pp import ParaphraseDetector
from .nli import Contradiction, Entailment
from .agreement import BLEU, BERTScore, AgreementNER, AgreementLLM

metric_mappings = {
    "bleu": BLEU,
    "bertscore": BERTScore,
    "ner": AgreementNER,
    "llm": AgreementLLM,
    "pp": ParaphraseDetector,
    "entailment": Entailment,
    "contradiction": Contradiction,
}


class Scorer:
    def __init__(self, metrics, aux_model=None):
        self.metrics = metrics
        self.metric_instances = []
        for _, metric in enumerate(metrics):
            if metric == "llm":
                if aux_model is None:
                    raise Exception("List of metrics contain option `llm` that requires argument `aux_model`.")
                else:
                    self.aux_model = aux_model
                    self.metric_instances.append(AgreementLLM(aux_model))
            else:
                self.metric_instances.append(metric_mappings[metric]())

    def score(self, input, outputs):
        pass


class PairwiseScorer(Scorer):

    def score(self, input, outputs, thresholds=None, binary=True, verbose=False):
        scores = {}
        if thresholds is None:
            thresholds = [0.5 for _ in self.metric_instances]

        for idx, metric in enumerate(self.metric_instances):
            if verbose:
                print("Calculating metric " + metric.name)
            metric_scores = 0
            for i, output_i in enumerate(outputs):
                for j, output_j in enumerate(outputs):
                    if i == j:
                        continue
                    if metric.name == "llm":
                        agreement_score = metric.score(input, output_i, output_j)
                    else:
                        agreement_score = metric.score(output_i, output_j)
                    if binary and agreement_score >= thresholds[idx]:
                        metric_scores += 1
                    elif binary == False:
                        metric_scores += agreement_score

            if (len(outputs) * (len(outputs) - 1)) == 0:
                scores[metric.name] = 0
            else:
                scores[metric.name] = (1 / (len(outputs) * (len(outputs) - 1))) * metric_scores

        return scores


class EntropyScorer(Scorer):

    def __init__(self, metrics, aux_model=None):
        self.surrogate_scorer = PairwiseScorer(metrics=metrics, aux_model=aux_model)

    def cluster(self, input, outputs, thresholds, verbose=False):
        """
        Organizes similar items from a list into clusters based on a similarity threshold.

        This function takes a list of items (`outputs`) and groups them into clusters.
        Each cluster contains items that are similar to each other based on a given similarity
        measure (`agreement_fn`). The similarity is compared to a specified threshold to
        determine if items belong in the same cluster.

        :param str input: A reference input used by `agreement_fn` to compare against items in `outputs`.
        :param list outputs: A list of items to be clustered.
        :param list thresholds: A list of threshold values for each metric.
        :return: A list of clusters, where each cluster is a list of items from 'outputs' that are similar to each other as per the 'agreement_fn' and above the 'threshold' value.
        :rtype: list

        Example:
            ```
            scorer.cluster(input_item, list_of_items)
            > [[item1, item2], [item3], [item4, item5, item6]]
            ```
        """
        if thresholds is None:
            thresholds = [0.5 for _ in self.surrogate_scorer.metric_instances]

        clusters = {}
        for idx, metric in enumerate(self.surrogate_scorer.metric_instances):
            if verbose:
                print("Obtaining clusters for metric " + metric.name)
            C = [[outputs[0]]]
            outs = outputs[1:]
            for i in range(len(outs)):
                STORED = False
                for j in range(len(C)):
                    s_c = C[j][0]
                    if metric.name == "llm":
                        left_score = metric.score(input, s_c, outs[i])
                        right_score = metric.score(input, outs[i], s_c)
                    else:
                        left_score = metric.score(s_c, outs[i])
                        right_score = metric.score(outs[i], s_c)

                    if left_score > thresholds[idx] and right_score > thresholds[idx]:
                        STORED = True
                        C[j].append(outs[i])
                if not STORED:
                    C.append([outs[i]])
            clusters[metric.name] = C
        
        return clusters

    def score(self, input, outputs, thresholds=None, verbose=False):

        self.clusters = self.cluster(input, outputs, thresholds, verbose)

        scores = {}
        for metric_name, cluster in self.clusters.items():
            cluster_probs = np.array([len(c) for c in cluster]) / sum([len(c) for c in cluster])
            scores[metric_name] = entropy(cluster_probs, base=2)

        return scores

    # def entropy_score(self, input, outputs, agreement_fn, threshold, _):
    #     """
    #     Calculates the entropy score for given outputs based on semantic clustering.

    #     Parameters:
    #         - input: The input based on which outputs were generated.
    #         - outputs (list): A list of generated outputs.
    #         - agreement_fn: A function to calculate agreement between outputs.
    #         - threshold: A threshold value for agreement.

    #     Returns:
    #         - float: The entropy score.
    #     """
    #     # TODO
    #     # Add exact score via entropy estimate through Monte Carlo
    #     clusters = semantic_clustering(input, outputs, agreement_fn, threshold)

    #     pk = np.array([len(c) for c in clusters]) / sum([len(c) for c in clusters])
    #     H = entropy(pk, base=2)
    #     return H

    # def pairwise_score(self, input, outputs, agreement_fn, threshold, binary=True):
    #     """
    #     Calculates pairwise agreement score for given outputs.

    #     Parameters:
    #         - input: The input based on which outputs were generated.
    #         - outputs (list): A list of generated outputs.
    #         - agreement_fn: Function to calculate agreement between two outputs.
    #         - threshold: A threshold value for considering an agreement.
    #         - binary (bool): If True, counts binary agreements; else adds up agreement scores.

    #     Returns:
    #         - float: The pairwise agreement score.
    #     """
    #     agreements = 0
    #     for i, output_i in enumerate(outputs):
    #         for j, output_j in enumerate(outputs):
    #             if i == j:
    #                 continue
    #             agreement_score = agreement_fn(input, output_i, output_j)
    #             if binary and agreement_score >= threshold:
    #                 agreements += 1
    #             elif binary == False:
    #                 agreements += agreement_score
    #     if (len(outputs) * (len(outputs) - 1)) == 0:
    #         return 0
    #     return (1 / (len(outputs) * (len(outputs) - 1))) * agreements

    # def score(self, input, outputs):
    #     """
    #     Calculates the consistency scores for the given outputs based on the scoring type.

    #     Parameters:
    #         - input: The input based on which outputs were generated.
    #         - outputs (list): A list of generated outputs.

    #     Returns:
    #         - dict: A dictionary of scores for each agreement function in agreements_list.
    #     """
    #     if self.scoring_type == "entropy":
    #         scorer = self.entropy_score
    #     elif self.scoring_type == "pairwise":
    #         scorer = self.pairwise_score
    #     else:
    #         raise Exception(f"scoring type '{self.scoring_type}' not available")

    #     con_scores = {}
    #     for name, threshold in self.agreements_list:
    #         fn = Agreement(name, self.aux_model).agreement_fn
    #         print("Getting score for ", name)
    #         con_scores[name] = scorer(input, outputs, fn, threshold, False)
    #     return con_scores
