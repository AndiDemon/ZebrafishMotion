#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def cmat_f1(cmat):
    """
    cmat: output of sklearn.metrics.confusion_matrix
               Pred
               0  1
    Truth 0 [[tn fp],
          1  [fn tp]]
    """

    tp = cmat[1, 1]
    fp = cmat[0, 1]
    fn = cmat[1, 0]

    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    else:
        return 2 * tp / denom


def cmat_recall(cmat) -> float:
    tp = cmat[1, 1]
    fn = cmat[1, 0]

    if tp == 0:
        return 0.0
    else:
        return tp / (tp + fn)


def cmat_specificity(cmat) -> float:
    tn = cmat[0, 0]
    fp = cmat[0, 1]

    if tn == 0:
        return 0.0
    else:
        return tn / (tn + fp)


def cmat_accuracy(cmat):
    tn = cmat[0, 0]
    tp = cmat[1, 1]

    return (tn + tp) / np.sum(cmat)
