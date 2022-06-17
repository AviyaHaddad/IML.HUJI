from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x_min, x_max = -1.2, 2
    X = np.random.rand(n_samples) * (x_max - x_min) + x_min
    eps = np.random.normal(0, noise, n_samples)
    f_x = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X), pd.Series(f_x + eps), 2 / 3)
    train_x = train_x.to_numpy().reshape(train_x.shape[0])
    test_x = test_x.to_numpy().reshape(test_x.shape[0])
    train_y = train_y.to_numpy().reshape(train_y.shape[0])
    test_y = test_y.to_numpy().reshape(test_y.shape[0])
    go.Figure([
        go.Scatter(name='True (noiseless) model', x=X, y=f_x, mode='markers'),
        go.Scatter(name='Train samples', x=train_x, y=train_y, mode='markers'),
        go.Scatter(name='Test samples', x=test_x, y=test_y, mode='markers')]) \
        .update_layout(title="Samples of the noiseless model, and the test and train samples with noise",
                       xaxis_title=r"Sample value",
                       yaxis_title=r"f(x) of a sample x").show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    max_pol_deg = 10
    k_range = list(range(max_pol_deg + 1))
    cv = 5
    train_errs = []
    val_errs = []
    for pol_deg in k_range:
        train_err, val_err = cross_validate(PolynomialFitting(pol_deg), train_x,
                                            train_y, mean_square_error, cv)
        train_errs.append(train_err)
        val_errs.append(val_err)

    go.Figure([
        go.Scatter(name='Train Errors', x=k_range, y=train_errs, mode='markers+lines'),
        go.Scatter(name='Validation Errors', x=k_range, y=val_errs, mode='markers+lines')]) \
        .update_layout(title="5-fold cross-validation for each of the polynomial degrees k = 0,1,...,10",
                       xaxis_title=r"Degree",
                       yaxis_title=r"Error").show()


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    lowest_val_err = int(np.argmin(val_errs))
    pol_fit = PolynomialFitting(lowest_val_err)
    pol_fit.fit(train_x, train_y)
    loss = mean_square_error(test_y, pol_fit.predict(test_x))
    print(lowest_val_err, loss)


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    X = np.array(X)
    y = np.array(y)
    train_x, train_y, test_x, test_y = X[:n_samples, ], y[:n_samples], X[n_samples:, ], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lamdas = np.linspace(0, 5, n_evaluations)
    ridge_train_losses = []
    ridge_val_losses = []
    lasso_train_losses = []
    lasso_val_losses = []
    for l in lamdas:
        r_train_loss, r_val_loss = cross_validate(RidgeRegression(l), train_x, train_y, mean_square_error)
        ridge_train_losses.append(r_train_loss)
        ridge_val_losses.append(r_val_loss)
        l_train_loss, l_val_loss = cross_validate(Lasso(l), train_x, train_y, mean_square_error)
        lasso_train_losses.append(l_train_loss)
        lasso_val_losses.append(l_val_loss)


    go.Figure([
        go.Scatter(name='Ridge train Errors', x=lamdas, y=ridge_train_losses, mode='markers+lines'),
        go.Scatter(name='Ridge validation Errors', x=lamdas, y=ridge_val_losses, mode='markers+lines'),
        go.Scatter(name='Lasso train Errors', x=lamdas, y=lasso_train_losses, mode='markers+lines'),
        go.Scatter(name='Lasso validation Errors', x=lamdas, y=lasso_val_losses, mode='markers+lines')]) \
        .update_layout(title="5-fold cross-validation with ridge and lasso algorithms, with different lamdas",
                       xaxis_title=r"Lamda",
                       yaxis_title=r"Error").show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_l_ridge = lamdas[int(np.argmin(ridge_val_losses))]
    best_l_lasso = lamdas[int(np.argmin(lasso_val_losses))]
    print("Best lamdas for Ridge and Lasso: ", best_l_ridge, best_l_lasso)
    least_square = LinearRegression()
    best_ridge = RidgeRegression(best_l_ridge)
    best_lasso = Lasso(best_l_lasso)
    best_ridge.fit(train_x, train_y)
    best_lasso.fit(train_x, train_y)
    least_square.fit(train_x, train_y)
    ridge_loss = mean_square_error(test_y, best_ridge.predict(test_x))
    lasso_loss = mean_square_error(test_y, best_lasso.predict(test_x))
    least_square_loss = mean_square_error(test_y, least_square.predict(test_x))
    print("Test errors of Ridge, Lasso and Squares regressiion: ", ridge_loss, lasso_loss, least_square_loss)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()

