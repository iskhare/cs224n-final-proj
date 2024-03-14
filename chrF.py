#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Maja Popovic

# The program is distributed under the terms
# of the GNU General Public Licence (GPL)

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Publications of results obtained through the use of original or
# modified versions of the software have to cite the authors by refering
# to the following publication:

# Maja Popović (2015).
# "chrF: character n-gram F-score for automatic MT evaluation".
# In Proceedings of the Tenth Workshop on Statistical Machine Translation (WMT15), pages 392–395
# Lisbon, Portugal, September 2015.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import math
import unicodedata

from collections import defaultdict
import string


def separate_characters(line):
    return list(line.strip().replace(" ", ""))


def separate_punctuation(line):
    words = line.strip().split()
    tokenized = []
    for w in words:
        if len(w) == 1:
            tokenized.append(w)
        else:
            lastChar = w[-1]
            firstChar = w[0]
            if lastChar in string.punctuation:
                tokenized += [w[:-1], lastChar]
            elif firstChar in string.punctuation:
                tokenized += [firstChar, w[1:]]
            else:
                tokenized.append(w)

    return tokenized


def ngram_counts(wordList, order):
    counts = defaultdict(lambda: defaultdict(float))
    nWords = len(wordList)
    for i in range(nWords):
        for j in range(1, order + 1):
            if i + j <= nWords:
                ngram = tuple(wordList[i:i + j])
                counts[j - 1][ngram] += 1

    return counts


def ngram_matches(ref_ngrams, hyp_ngrams):
    matchingNgramCount = defaultdict(float)
    totalRefNgramCount = defaultdict(float)
    totalHypNgramCount = defaultdict(float)

    for order in ref_ngrams:
        for ngram in hyp_ngrams[order]:
            totalHypNgramCount[order] += hyp_ngrams[order][ngram]
        for ngram in ref_ngrams[order]:
            totalRefNgramCount[order] += ref_ngrams[order][ngram]
            if ngram in hyp_ngrams[order]:
                matchingNgramCount[order] += min(ref_ngrams[order][ngram], hyp_ngrams[order][ngram])

    return matchingNgramCount, totalRefNgramCount, totalHypNgramCount


def ngram_precrecf(matching, reflen, hyplen, beta):
    ngramPrec = defaultdict(float)
    ngramRec = defaultdict(float)
    ngramF = defaultdict(float)

    factor = beta ** 2

    for order in matching:
        if hyplen[order] > 0:
            ngramPrec[order] = matching[order] / hyplen[order]
        else:
            ngramPrec[order] = 1e-16
        if reflen[order] > 0:
            ngramRec[order] = matching[order] / reflen[order]
        else:
            ngramRec[order] = 1e-16
        denom = factor * ngramPrec[order] + ngramRec[order]
        if denom > 0:
            ngramF[order] = (1 + factor) * ngramPrec[order] * ngramRec[order] / denom
        else:
            ngramF[order] = 1e-16

    return ngramF, ngramRec, ngramPrec


def computeChrF(refs, pred, nworder=2, ncorder=6, beta=2.0):
    """
    Calculate character n-gram F scores, by also checking the word n-grams, a.k.a. chrf++.
    :param refs: All reference sentences - ground truth
    :param pred: Submitted sentence
    :param nworder: N-gram for word
    :param ncorder: N-gram for character
    :param beta: Fixed to 2.0
    :return: ChrF++ score
    """
    norder = float(nworder + ncorder)

    maxF = 0.0

    hypNgramCounts = ngram_counts(separate_punctuation(pred), nworder)
    hypChrNgramCounts = ngram_counts(separate_characters(pred), ncorder)

    # going through multiple references
    for ref in refs:
        refNgramCounts = ngram_counts(separate_punctuation(ref), nworder)
        refChrNgramCounts = ngram_counts(separate_characters(ref), ncorder)

        # number of overlapping n-grams, total number of ref n-grams, total number of hyp n-grams
        matchingNgramCounts, totalRefNgramCount, totalHypNgramCount = ngram_matches(refNgramCounts, hypNgramCounts)
        matchingChrNgramCounts, totalChrRefNgramCount, totalChrHypNgramCount = ngram_matches(refChrNgramCounts,
                                                                                             hypChrNgramCounts)

        # n-gram f-scores, recalls and precisions
        ngramF, ngramRec, ngramPrec = ngram_precrecf(matchingNgramCounts, totalRefNgramCount, totalHypNgramCount,
                                                     beta)
        chrNgramF, chrNgramRec, chrNgramPrec = ngram_precrecf(matchingChrNgramCounts, totalChrRefNgramCount,
                                                              totalChrHypNgramCount, beta)


        sentF = (sum(chrNgramF.values()) + sum(ngramF.values())) / norder

        if sentF > maxF:
            maxF = sentF

    return maxF