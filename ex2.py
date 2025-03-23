import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def simulate_seasonal_arima(phi=[1], d=1, theta=[1], Phi=[1], D=1, Theta=[1], s=12, n=500, label=''):
    """
    Simulates a seasonal ARIMA (p, d, q) × (P, D, Q)_s model.

    Parameters:
    p, d, q  : Non-seasonal ARIMA parameters
    P, D, Q  : Seasonal ARIMA parameters
    s        : Seasonal period
    n        : Number of time steps

    Returns:
    Simulated time series data.
    """
    p = len(phi)
    q = len(theta)
    P = len(Phi)
    Q = len(Theta)
    
    # Define AR and MA components
    phi = [1] + phi  # Non-seasonal AR coefficients
    theta = [1] + theta  # Non-seasonal MA coefficients

    Phi = [1] + Phi  # Seasonal AR coefficients
    Theta = [1] + Theta  # Seasonal MA coefficients

    # Generate non-seasonal ARMA process
    arma_process = ArmaProcess(phi, theta)
    y = arma_process.generate_sample(nsample=n + (D * s))  # Extra points for differencing

    # Apply differencing
    def difference(series, interval=1):
        return series[interval:] - series[:-interval]

    # Apply non-seasonal differencing (d)
    for _ in range(d):
        y = difference(y)

    # Apply seasonal differencing (D)
    for _ in range(D):
        y = difference(y, s)

    # Generate seasonal ARMA component and add it
    seasonal_process = ArmaProcess(Phi, Theta)
    y_seasonal = seasonal_process.generate_sample(nsample=len(y))
    y += y_seasonal  # Combine seasonal + non-seasonal parts

    # Plot the time series
    plt.figure(figsize=(12, 5))
    plt.plot(y)
    plt.title(f'{label} Simulated Seasonal ARIMA ({p},{d},{q}) × ({P},{D},{Q})_{s} phi={phi[1:]}, theta={theta[1:]}, Phi={Phi[1:]}, Theta={Theta[1:]}')
    plt.xlabel('Time')
    plt.ylabel('Y_t')
    #plt.show()
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
    #plt.show()

    return y

# Example usage:
# 2.1
simulate_seasonal_arima(phi=[0.6], d=0, theta=[], Phi=[], D=0, Theta=[], s=12, n=1200, label='Example 2.1')
# 2.2
simulate_seasonal_arima(phi=[], d=0, theta=[], Phi=[-0.9], D=0, Theta=[], s=12, n=1200, label='Example 2.2')

# 2.3
simulate_seasonal_arima(phi=[0.9], d=0, theta=[], Phi=[], D=0, Theta=[-0.7], s=12, n=1200, label='Example 2.3')

# 2.4
simulate_seasonal_arima(phi=[-0.6], d=0, theta=[], Phi=[-0.8], D=0, Theta=[], s=12, n=1200,  label='Example 2.4')

# 2.5
simulate_seasonal_arima(phi=[], d=0, theta=[0.4], Phi=[], D=0, Theta=[-0.8], s=12, n=1200, label='Example 2.5')

# 2.6
simulate_seasonal_arima(phi=[], d=0, theta=[-0.4], Phi=[0.7], D=0, Theta=[], s=12, n=1200, label='Example 2.6')

#simulate_seasonal_arima(0,0,0,1,0,0,12, 1200)

