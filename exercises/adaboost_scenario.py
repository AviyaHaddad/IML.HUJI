import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    train_loss, test_loss = [], []
    iterations = np.arange(n_learners)
    for i in iterations:
        train_loss.append(adaboost.partial_loss(train_X, train_y, i + 1))
        test_loss.append(adaboost.partial_loss(test_X, test_y, i + 1))
    go.Figure([go.Scatter(x=iterations, y=train_loss, mode='markers+lines', name=r'Train loss'),
               go.Scatter(x=iterations, y=test_loss, mode='markers+lines', name=r'Test loss')],
              layout=go.Layout(
                  title=r"$\text{Training and test errors as a function of the number of fitted learners}$",
                  xaxis_title="Iterations", yaxis_title="Loss", height=600)).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"{t} iterations" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        def x_predict(x):
            return adaboost.partial_predict(x, t)
        fig.add_traces([decision_surface(x_predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, symbol="x", colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=0.5)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"Decision boundary obtained by using the the ensemble up to t iteration",
                      margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 3: Decision surface of best performing ensemble
    min_loss_iterations = int(np.argmin(test_loss)) + 1
    loss_val = adaboost.partial_loss(test_X, test_y, min_loss_iterations)
    fig = make_subplots(rows=1, cols=1, subplot_titles=[rf"{min_loss_iterations} iterations, loss = {loss_val}"],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    def x_best_predict(x):
        return adaboost.partial_predict(x, min_loss_iterations)
    fig.add_traces([decision_surface(x_best_predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y, symbol="x", colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=0.5)))]).\
        update_layout(title=rf"Decision surface of the ensemble which achieved the lowes test error",
                      margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 4: Decision surface with weighted samples
    if noise == 0:
        D = adaboost.D_ / np.max(adaboost.D_) * 20
    else:
        D = adaboost.D_ / np.max(adaboost.D_) * 5
    fig = make_subplots(rows=1, cols=1, horizontal_spacing=0.01, vertical_spacing=.03)
    def x_250_predict(x):
        return adaboost.partial_predict(x, 250)
    fig.add_traces([decision_surface(x_250_predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=train_y, symbol="x", size=D, colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=0.5)))]).\
        update_layout(title=rf"The training set with a point size proportional to it’s weight ",
                      margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False).show()



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
