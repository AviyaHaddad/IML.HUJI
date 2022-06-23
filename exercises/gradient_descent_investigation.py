import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import IMLearn
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error
from sklearn.metrics import roc_curve, auc

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    vals_list = []
    weights_list = []

    def callback(solver, weights, val, grad, t, eta, delta):
        vals_list.append(val)
        weights_list.append(weights)

    return callback, vals_list, weights_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    L1_vals = []
    L2_vals = []
    for lr in etas:
        lr = FixedLR(lr)
        L1_callback, L1_vals_list, L1_weights_list = get_gd_state_recorder_callback()
        f1 = L1(init)
        gd1 = GradientDescent(learning_rate=lr, callback=L1_callback)
        gd1.fit(f1, np.nan, np.nan)
        L1_weights_list = np.array(L1_weights_list)
        L1_vals_list = np.array(L1_vals_list)
        L1_vals.append(L1_vals_list)
        plot_descent_path(L1, L1_weights_list, f"for L1 with eta={lr.lr_step()}").show()

        L2_callback, L2_vals_list, L2_weights_list = get_gd_state_recorder_callback()
        f2 = L2(init)
        gd2 = GradientDescent(learning_rate=lr, callback=L2_callback)
        gd2.fit(f2, np.nan, np.nan)
        L2_weights_list = np.array(L2_weights_list)
        L2_vals_list = np.array(L2_vals_list)
        L2_vals.append(L2_vals_list)
        plot_descent_path(L2, L2_weights_list, f"for L2 with eta={lr.lr_step()}").show()

    iterations = np.arange(1000)
    go.Figure([
        go.Scatter(name=f'eta={etas[0]}', x=iterations, y=L1_vals[0], mode='markers+lines'),
        go.Scatter(name=f'eta={etas[1]}', x=iterations, y=L1_vals[1], mode='markers+lines'),
        go.Scatter(name=f'eta={etas[2]}', x=iterations, y=L1_vals[2], mode='markers+lines'),
        go.Scatter(name=f'eta={etas[3]}', x=iterations, y=L1_vals[3], mode='markers+lines')]) \
        .update_layout(title=f"GD loss for L1 with different etas",
                       xaxis_title=r"Iteration",
                       yaxis_title=r"Loss").show()
    go.Figure([
        go.Scatter(name=f'eta={etas[0]}', x=iterations, y=L2_vals[0], mode='markers+lines'),
        go.Scatter(name=f'eta={etas[1]}', x=iterations, y=L2_vals[1], mode='markers+lines'),
        go.Scatter(name=f'eta={etas[2]}', x=iterations, y=L2_vals[2], mode='markers+lines'),
        go.Scatter(name=f'eta={etas[3]}', x=iterations, y=L2_vals[3], mode='markers+lines')]) \
        .update_layout(title=f"GD loss for L2 with different etas",
                       xaxis_title=r"Iteration",
                       yaxis_title=r"Loss").show()

    print(np.min(L1_vals[0]))
    print(np.min(L1_vals[1]))
    print(np.min(L1_vals[2]))
    print(np.min(L1_vals[3]))
    print(np.min(L2_vals[0]))
    print(np.min(L2_vals[1]))
    print(np.min(L2_vals[2]))
    print(np.min(L2_vals[3]))



def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    L1_vals = []
    L1_weights = []
    for g in gammas:
        lr = ExponentialLR(eta, g)
        L1_callback, L1_vals_list, L1_weights_list = get_gd_state_recorder_callback()
        f1 = L1(init)
        gd1 = GradientDescent(learning_rate=lr, callback=L1_callback)
        gd1.fit(f1, np.nan, np.nan)
        L1_weights_list = np.array(L1_weights_list)
        L1_weights.append(L1_weights_list)
        L1_vals_list = np.array(L1_vals_list)
        L1_vals.append(L1_vals_list)

    # Plot algorithm's convergence for the different values of gamma
    iterations = np.arange(1000)
    go.Figure([
        go.Scatter(name=f'gamma={gammas[0]}', x=iterations, y=L1_vals[0], mode='markers+lines'),
        go.Scatter(name=f'gamma={gammas[1]}', x=iterations, y=L1_vals[1], mode='markers+lines'),
        go.Scatter(name=f'gamma={gammas[2]}', x=iterations, y=L1_vals[2], mode='markers+lines'),
        go.Scatter(name=f'gamma={gammas[3]}', x=iterations, y=L1_vals[3], mode='markers+lines')]) \
        .update_layout(title=f"GD loss for L1 with different gammas for exponential decay",
                       xaxis_title=r"Iteration",
                       yaxis_title=r"Loss").show()

    # Plot descent path for gamma=0.95
    plot_descent_path(L1, L1_weights[1], f"for L1 with exponentioal decay with gamma=0.95").show()
    print(np.min(L1_vals[3]))
    print(np.min(L1_vals[2]))
    print(np.min(L1_vals[1]))


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    lg = LogisticRegression()
    lg.fit(X_train, y_train)
    y_prob = lg.predict_proba(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color="rgb(215,48,39)",
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()
    opt_a = thresholds[np.argmax(tpr - fpr)]
    print(opt_a)
    lg.alpha_ = opt_a
    print(lg.loss(X_test, y_test))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    train_scores, val_scores = [], []
    # # Picking best lambda for L1
    gd = GradientDescent(max_iter=2000, learning_rate=FixedLR(1e-4))
    for l in lambdas:
        lg = LogisticRegression(solver=gd, penalty="l1", lam=l)
        train_score, val_score = cross_validate(lg, X_train, y_train, misclassification_error, cv=2)
        train_scores.append(train_score)
        val_scores.append(val_score)
    min_score_i = np.argmin(val_scores)
    best_l = lambdas[min_score_i]
    best_lg = LogisticRegression(penalty="l1", lam=best_l)
    best_lg.fit(X_train, y_train)
    print(best_l, best_lg.loss(X_test, y_test))

    # # Picking best lambda for L2
    train_scores, val_scores = [], []
    for l in lambdas:
        lg = LogisticRegression(solver=gd, penalty="l2", lam=l)
        train_score, val_score = cross_validate(lg, X_train, y_train, misclassification_error, cv=2)
        train_scores.append(train_score)
        val_scores.append(val_score)
    min_score_i = np.argmin(val_scores)
    best_l = lambdas[min_score_i]
    best_lg = LogisticRegression(penalty="l2", lam=best_l)
    best_lg.fit(X_train, y_train)
    print(best_l, best_lg.loss(X_test, y_test))



if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
