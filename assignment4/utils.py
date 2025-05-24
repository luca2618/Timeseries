import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.optimize import minimize




def calculate_aic_bic(y_true, y_pred, k):
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return aic, bic


def return_1d_params(params):
    A = params[0]
    B = np.array(params[1:4]).reshape((1, 3))  # shape (1, 3)
    #sigma_1 = params[4]**2  # Q - system covariance (positive definite)
    sigma_1 = params[4] # Q - system covariance (positive definite)
    C = params[5]

    #sigma_2 = params[6]**2  # R - observation covariance (positive definite)
    sigma_2 = params[6] # R - observation covariance (positive definite)
    return A, B, sigma_1, C, sigma_2


def kf_log_likelihood_1d(params, df, return_all=False):
    # Extract parameters
    A, B, sigma_1, C, sigma_2 = return_1d_params(params)

    # Data
    Y = df['Y'].to_numpy()  # shape (T, 1)
    # print("Y shape:", Y.shape)
    # print(Y)
    U = df[['Ta', 'S', 'I']].to_numpy()  # shape (T, 3)
    Tn = len(Y)

    # Initialization
    x_est = Y[0]  # shape (1, 1)
    P_est = np.eye(1)*0.1
    log_likelihood = 0

    x_filtered = []
    y_predicted = []

    for t in range(Tn):
        # Prediction step
        x_pred = A * x_est + B @ U[t]  # shape (1,1)
        P_pred = A * P_est * A + sigma_1

        # Innovation step
        y_pred = C * x_pred

        y_predicted.append(y_pred.item())

        S_t = C * P_pred * C + sigma_2
        innov = Y[t] - y_pred

        # Log-likelihood
        log_likelihood -= 0.5 * (np.log(2 * np.pi * S_t) + (innov ** 2) / S_t)

        # Update step
        K_t = P_pred * C / S_t
        x_est = x_pred + K_t * innov
        P_est = (1 - K_t * C) * P_pred
        x_filtered.append(x_est.item())
    if return_all:
        return -log_likelihood, np.array(y_predicted)
    else:
        return -log_likelihood


def estimate_dt_1d(df, lower=None, upper=None):
    def neg_log_likelihood(par):
        return kf_log_likelihood_1d(par, df)
    
    np.random.seed(0)
    start_par = np.random.uniform(-1, 1, size=7)
    #var are [A, B1, B2, B3, sqrt(sigma_1), C, sqrt(sigma_2)]
    lower = [-5, -10, -10, -10, 0.001, -5, 0.001]
    upper = [10, 10, 10, 10, 10, 5, 5]
    result = minimize(
        neg_log_likelihood,
        x0=start_par,
        bounds=list(zip(lower, upper)) if lower is not None else None,
        method="L-BFGS-B",
        options={"maxiter": 1000, "disp": True}
    )
    return result

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plots_and_stats(data, y_predicted, k=7):

    # 2 by 2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    residual = data["Y"] - y_predicted

    plot_acf(residual, ax=axes[0,0], lags=50, title="ACF of Residuals")
    plot_pacf(residual, ax=axes[0,1], lags=50, title="PACF of Residuals")
    # residual plot
    
    axes[1,0].plot(data["time"], residual, label="Residuals", color='tab:blue')
    axes[1,0].set_title("Residuals")
    axes[1,0].set_xlabel("Time")
    axes[1,0].set_ylabel("Residuals")


    #residual histogram
    # plt.hist(residual, bins=30, alpha=0.5, color='tab:blue', density=True)
    # plt.title("Histogram of Residuals")
    # plt.show()

    # QQ-plot
    import scipy.stats as stats
    import statsmodels.api as sm
    sm.qqplot(residual, line='s', ax=axes[1,1])
    axes[1,1].set_title("QQ-plot of Residuals")

    # Calculate AIC and BIC
    aic, bic = calculate_aic_bic(data["Y"], y_predicted, k=k)
    print(f"AIC: {aic}, BIC: {bic}")





def return_2d_params(params):
    A = np.array(params[0:4]).reshape(2, 2)
    B = np.array(params[4:10]).reshape((2,3)) 
     # Cholesky parameters for Q (sigma_1)
    l11 = params[10]  # ensure positivity
    l21 = params[11]
    l22 = params[12] # ensure positivity
    L = np.array([[l11, 0],
                  [l21, l22]])
    sigma_1 = L @ L.T
    C = np.array(params[13:15]).reshape(1, 2)
    sigma_2 = params[15]**2 # R - observation covariance (positive definite)
    return A, B, sigma_1, C, sigma_2


def kf_log_likelihood_2d(params, df, return_all=False):
    # Extract parameters
    A, B, sigma_1, C, sigma_2 = return_2d_params(params)

    # Data
    Y = df['Y'].to_numpy()  # shape (T, 1)
    # print("Y shape:", Y.shape)
    # print(Y)
    U = df[['Ta', 'S', 'I']].to_numpy()  # shape (T, 3)
    Tn = len(Y)

    # Initialization
    x_est = np.array([20,20])  # shape (1, 1)
    P_est = np.eye(2)
    log_likelihood = 0

    x_filtered = []
    y_predicted = []

    for t in range(Tn):
        # Prediction step
        # print("U[t]:", U[t])
        # print("x_est:", x_est)
        x_pred = A @ x_est + B @ U[t]
        P_pred = A @ P_est @ A.T + sigma_1

        # Innovation
        y_pred = C @ x_pred
        # print("y_pred:", y_pred)
        # print("C:", C)
        # print("x_pred:", x_pred)
       
        S_t = (C @ P_pred @ C.T).item() + sigma_2
        if S_t < 0:
            # print("A:", A)
            # print("sigma_1:", sigma_1)
            # print("S_t:", S_t)
            # print("Pred:", P_pred)
            return np.inf  # Return infinity if S_t is negative to avoid log of negative number
            #raise ValueError("S_T is negative, check your parameters.:"+ str(S_t))
        #S_t = max(S_t, 1e-3)  # Ensure positive definiteness
        innov = Y[t] - y_pred.item()

        # Log-likelihood
        log_likelihood -= 0.5 * (np.log(2 * np.pi * S_t) + (innov ** 2) / S_t)

        # Update
        K_t = (P_pred @ C.T / S_t).flatten()  # Kalman gain
        # print("K_t:", K_t)
        x_est = x_pred + K_t * innov
        P_est = (np.eye(2) - K_t[:, None] @ C) @ P_pred

        x_filtered.append(x_est.copy())
        y_predicted.append(y_pred.item())

    if return_all:
        return -log_likelihood, np.array(x_filtered), np.array(y_predicted)
    else:
        return -log_likelihood
    
def estimate_dt_2d(df, lower=None, upper=None):
    def neg_log_likelihood(par):
        return kf_log_likelihood_2d(par, df)
    
    np.random.seed(0)
    start_par = np.array([0.1]*16)
    #var are [A           B        sigma_1     C      sigma_2]
    lower = [-5]*4 + [-5]*6 + [-5]*3 + [-5]*2 + [-5]
    upper = [5]*4 + [5]*6 + [5]*3 + [5]*2 + [5]
    result = minimize(
        neg_log_likelihood,
        x0=start_par,
        bounds=list(zip(lower, upper)) if lower is not None else None,
        method="L-BFGS-B",
        options={"maxiter": 1000, "disp": True}
    )
    return result