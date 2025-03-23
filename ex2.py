from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def simulate_seasonal_arima_sarimax(phi, d, theta, Phi, D, Theta, s, n, label=''):
    """
    Simulates a seasonal ARIMA (p, d, q) × (P, D, Q)_s model using SARIMAX.

    Parameters:
    phi, d, theta  : Non-seasonal ARIMA parameters
    Phi, D, Theta  : Seasonal ARIMA parameters
    s              : Seasonal period
    n              : Number of time steps
    label          : Label for plots and output files

    Returns:
    Simulated time series data.
    """
    # Define SARIMAX model
    model = SARIMAX(
        endog=np.zeros(n),  # Placeholder for endogenous variable
        order=(len(phi), d, len(theta)),
        seasonal_order=(len(Phi), D, len(Theta), s),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Simulate data
    params = np.r_[phi, theta, Phi, Theta, 1.0]  # Combine all parameters
    print(params.shape)
    print(params)
    y = model.simulate(params, n)

    p = len(phi)
    q = len(theta)
    P = len(Phi)
    Q = len(Theta)

    # Plot the time series
    plt.figure(figsize=(12, 5))
    plt.plot(y)
    plt.title(f'{label} Simulated Seasonal ARIMA ({p},{d},{q}) × ({P},{D},{Q})_{s} phi={phi[1:]}, theta={theta[1:]}, Phi={Phi[1:]}, Theta={Theta[1:]}')
    plt.xlabel('Time')
    plt.ylabel('Y_t')
    plt.savefig(f'{label}.png')
    plt.close()

    # Plot ACF and PACF
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(y, lags=24, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)')
    plot_pacf(y, lags=24, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    plt.tight_layout()
    plt.savefig(f'{label}_acf_pacf.png')
    plt.close()

    return y

# Example usage:
# 2.1
simulate_seasonal_arima_sarimax(phi=[0.6], d=0, theta=[], Phi=[], D=0, Theta=[], s=12, n=1200, label='Example 2.1')
# 2.2
simulate_seasonal_arima_sarimax(phi=[], d=0, theta=[], Phi=[-0.9], D=0, Theta=[], s=12, n=1200, label='Example 2.2')

# 2.3
simulate_seasonal_arima_sarimax(phi=[0.9], d=0, theta=[], Phi=[], D=0, Theta=[-0.7], s=12, n=1200, label='Example 2.3')

# 2.4
simulate_seasonal_arima_sarimax(phi=[-0.6], d=0, theta=[], Phi=[-0.8], D=0, Theta=[], s=12, n=1200,  label='Example 2.4')

# 2.5
simulate_seasonal_arima_sarimax(phi=[], d=0, theta=[0.4], Phi=[], D=0, Theta=[-0.8], s=12, n=1200, label='Example 2.5')

# 2.6
simulate_seasonal_arima_sarimax(phi=[], d=0, theta=[-0.4], Phi=[0.7], D=0, Theta=[], s=12, n=1200, label='Example 2.6')

#simulate_seasonal_arima_sarimax(0,0,0,1,0,0,12, 1200)

