import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from chrF import computeChrF
from CharacTER import cer


def bleu_score(refs, pred):
    """
    Return bleu-2 score
    :param refs: reference sentences (multiple translations possible)
    :param pred: prediction
    :param order: n in n-gram
    :return: bleu-2
    """
    ref_tok_lst = [ref.split() for ref in refs]
    pred_tok = pred.split()
    cc = SmoothingFunction()

    n = 2
    ws = tuple(1/n for i in range(n))
    sc = sentence_bleu(ref_tok_lst, pred_tok, weights = ws, smoothing_function=cc.method3)
    return float(sc)


def chrfplus_score(refs, pred):
    """
    :param refs: multiple reference sentences
    :param pred: predicted
    :return: chrf-3 score
    """
    chrf_3 = computeChrF(refs, pred, nworder=0, ncorder=3)
    return chrf_3


def cter_score(refs, pred):
    """
    Calculate the characTER scores for each reference - predicted sentence
    :param refs: multiple reference sentences
    :param pred: predicted
    :return: the maximum of these scores (minimum TER)
    """
    min_cter_cost = 1.
    pred = pred.split()
    for ref in refs:
        ref = ref.split()
        cter_cost = cer(pred, ref)
        if cter_cost < min_cter_cost:
            min_cter_cost = cter_cost

    return 1.0 - min_cter_cost


def em_score(refs, pred):
    """
    Calculate the exact match score for each reference - predicted sentence
    :param refs: multiple reference sentences
    :param pred: predicted
    :return: the maximum of these scores
    """
    for ref in refs:
        if ref == pred:
            return 1.
    # else no match
    return 0.