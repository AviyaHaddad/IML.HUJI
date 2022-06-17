from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator
from IMLearn.utils import split_train_test
import pandas as pd


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    fold_size = int(np.round(X.shape[0] / cv))
    folds_x = []
    folds_y = []
    for i in range(cv):
        if (i + 1) * fold_size > X.shape[0]:
            folds_x.append(X[i * fold_size:])
            folds_y.append(y[i * fold_size:])
        else:
            folds_x.append(X[i * fold_size:(i + 1) * fold_size])
            folds_y.append(y[i * fold_size:(i + 1) * fold_size])
    train_score = []
    val_score = []
    for k in range(cv):
        new_train_x = np.concatenate([folds_x[i] for i in range(cv) if i != k], axis=0)
        new_train_y = np.concatenate([folds_y[i] for i in range(cv) if i != k], axis=0)
        val_x = folds_x[k]
        val_y = folds_y[k]
        estimator.fit(new_train_x, new_train_y)
        train_score.append(scoring(new_train_y, estimator.predict(new_train_x)))
        val_score.append(scoring(val_y, estimator.predict(val_x)))
    return sum(train_score) / cv, sum(val_score) / cv