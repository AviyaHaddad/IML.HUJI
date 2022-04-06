from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    univar_gaus_est = UnivariateGaussian()
    univar_gaus_est.fit(X)
    print(f"({univar_gaus_est.mu_}, {univar_gaus_est.var_})\n")

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(np.int)
    estimated_mean = []
    for m in ms:
        estimated_mean.append(np.mean(X[:m]))

    go.Figure([go.Scatter(x=ms, y=np.abs(np.array(estimated_mean) - 10),
                          mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Estimation of Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$\\text{number of samples}$",
                               yaxis_title="r$|\hat\mu - \mu|$",
                               height=600)).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    go.Figure([go.Scatter(x=X, y=univar_gaus_est.pdf(X),
                          mode='markers', name=r'$\text{PDF}$')],
              layout=go.Layout(title=r"$\text{Empirical PDF function under the fitted model}$",
                               xaxis_title="$\\text{Sample value}$",
                               yaxis_title="$\\text{Sample PDF's}$",
                               height=600)).show()




def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, 1000)
    multivar_gaus_est = MultivariateGaussian()
    multivar_gaus_est.fit(X)
    print(f"mu: {multivar_gaus_est.mu_}\n cov: {multivar_gaus_est.cov_}\n")

    # Question 5 - Likelihood evaluation
    foos_vals = np.linspace(-10, 10, 200)
    iterable = (MultivariateGaussian.log_likelihood(np.array([f1, 0, f3, 0]), cov, X)
                for f1 in foos_vals for f3 in foos_vals)
    log_likelihood_mat = np.fromiter(iterable, "float").reshape((200, 200))
    go.Figure(go.Heatmap(x=foos_vals, y=foos_vals, z=log_likelihood_mat, colorbar={"title":"Log-likelihood"}),
              layout=go.Layout(title="$\\text{Heat-map of the Log-Likelihood for models with the expectation [f1, 0, f3, 0] while f1 and f3 are values between -10 to 10}$",
                               xaxis_title="$f3$", yaxis_title="$f1$", width=1100, height=600)).show()


    # Question 6 - Maximum likelihood
    max_log_likelihood_f1_i = np.argmax(log_likelihood_mat, axis=0)[0]
    max_log_likelihood_f3_i = np.argmax(log_likelihood_mat, axis=1)[0]
    f1_val = np.round(foos_vals[max_log_likelihood_f1_i], 3)
    f3_val = np.round(foos_vals[max_log_likelihood_f3_i], 3)
    max_log_likelihood = np.round(log_likelihood_mat[max_log_likelihood_f1_i, max_log_likelihood_f3_i], 3)
    print(f"Max log likelihood value: {max_log_likelihood}, with f1: {f1_val}, and f3: {f3_val}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
    x = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
                  -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    print(UnivariateGaussian.log_likelihood(10, 1, x))