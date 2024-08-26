import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from scipy.stats import chi2


def multivariate_acf(data, nlags=10):
    """
    Calculate the multivariate ACF for each pair of time series in the data.
    
    Parameters:
    - data: A 2D numpy array or pandas DataFrame where each column is a different time series.
    - nlags: Number of lags to calculate ACF for.
    
    Returns:
    - acf_matrix: A dictionary where keys are lag values and values are the autocorrelation matrices.
    """
    n_series = data.shape[1]
    acf_matrix = {}
    
    for lag in range(nlags + 1):
        acf_mat = np.zeros((n_series, n_series))
        for i in range(n_series):
            for j in range(n_series):
                series_i = data[:, i]
                series_j = data[:, j]
                acf_ij = np.corrcoef(series_i[:-lag or None], series_j[lag:])[0, 1]
                if np.isnan(acf_ij):
                    acf_ij = 0
                acf_mat[i, j] = acf_ij
        acf_matrix[lag] = acf_mat
    
    return acf_matrix


def plot_acf_heatmaps(acf_matrices):
    n_lags = len(acf_matrices)
    n_cols = 3  # Number of charts per row
    n_rows = (n_lags + n_cols - 1) // n_cols  # Calculate number of rows needed

    plt.figure(figsize=(12, 4 * n_rows))  # Adjust figure size to accommodate all plots

    for i, (lag, acf_matrix) in enumerate(acf_matrices.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.heatmap(acf_matrix, annot=False, cmap='coolwarm', center=0, cbar=True)
        plt.title(f'ACF Matrix at Lag {lag}')
    
    plt.tight_layout()  # Adjust subplots to fit in the figure area without overlapping
    plt.show()


def multivariate_portmanteau_test(data, lags=10):
    """
    Perform a multivariate Portmanteau test on the residuals of a VAR model.
    NOTE: The number of observations is too small compared to the number of lags to perform this test.
    
    Parameters:
    - data: A 2D numpy array or pandas DataFrame where each column is a different time series.
    - lags: The number of lags to include in the test.
    
    Returns:
    - q_stat: The Portmanteau statistic.
    - p_value: The p-value of the test.
    """
    # Fit a VAR model
    model = VAR(data)
    results = model.fit(maxlags=lags, ic='aic')
    
    # Calculate the Portmanteau statistic
    q_stat = 0
    for lag in range(1, lags + 1):
        # Calculate the residual autocorrelation matrices
        residual_acf = multivariate_acf(results.resid, nlags=lag)[lag]
        q_stat += np.trace(np.dot(residual_acf.T, residual_acf)) / (results.nobs - lag)
    
    # Adjust Q-stat by number of lags and variables
    q_stat *= results.nobs * (results.nobs + 2)
    q_stat = q_stat / (results.nobs - lags)
    
    # Degrees of freedom for the chi-squared test
    dof = (data.shape[1] ** 2) * lags
    
    # Calculate the p-value
    p_value = 1 - chi2.cdf(q_stat, df=dof)
    
    return q_stat, p_value

