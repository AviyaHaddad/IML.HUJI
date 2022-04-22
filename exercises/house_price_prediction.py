from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename).dropna().drop_duplicates().drop(["id", "lat", "long", "date"], 1)
    data = data[(data["price"] > 0) & (data["sqft_living15"] > 0) & (data["sqft_lot15"] > 0) &
                (data["sqft_above"] > 0) & (data["bedrooms"] >= 0) & (data["bathrooms"] >= 0) &
                (data["floors"] > 0)]
    # change the zipcode to categorical features
    data.zipcode = data.zipcode.astype(int)
    data = pd.get_dummies(data, prefix="zipcode", columns=["zipcode"], drop_first=True)
    # change the build-year to categorical features
    data.yr_built = (data.yr_built // 10).astype(int)
    data = pd.get_dummies(data, prefix="dec_built", columns=["yr_built"], drop_first=True)
    data.insert(loc=0, column="intercept", value=1, allow_duplicates=True)
    return pd.DataFrame(data.drop("price", 1)), pd.Series(data["price"])


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    beneficial_f = X.sqft_living
    other_f = X.sqft_lot15
    correlation = np.abs((np.cov(beneficial_f, y))[0, 1] /
                         (np.std(beneficial_f) * np.std(y)))
    plt.scatter(beneficial_f, y)
    plt.title(f'Pearson Correlation = {correlation}')
    plt.xlabel('sqft_living')
    plt.ylabel('Price')
    plt.savefig(f"{output_path}/feature_evaluation_1")
    correlation = np.abs((np.cov(other_f, y))[0, 1] /
                         (np.std(other_f) * np.std(y)))
    plt.scatter(other_f, y)
    plt.title(f'Pearson Correlation = {correlation}')
    plt.xlabel('sqft_lot15')
    plt.ylabel('Price')
    plt.savefig(f"{output_path}/feature_evaluation_2")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "ex2_graphs")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lin_reg = LinearRegression()
    means = np.ndarray((91, 1))
    stds = np.ndarray((91, 1))
    for i in range(10, 101):
        loss_array = np.ndarray((10, 1))
        for j in range(0, 10):
            set_i = train_X[:int(train_X.shape[0] * (i / 100))]
            res_i = train_y[:int(train_y.shape[0] * (i / 100))]
            lin_reg.fit(set_i, res_i)
            loss_array[j] = lin_reg.loss(test_X.to_numpy(), test_y.to_numpy())
        means[i] = loss_array.mean()
        stds[i] = loss_array.std()
    plt.plot(means, 'ro-')
    plt.title('MSE through learning')
    plt.xlabel('Percent of the training set')
    plt.ylabel('MSE value')
    plt.show()


