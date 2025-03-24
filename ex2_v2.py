from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def simulate_seasonal_arima_sarimax(phi, theta, Phi, Theta, p, d, q, P, D, Q, s, n=1200):
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
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        trend='n',
        enforce_stationarity=True,
        enforce_invertibility=True
    )

    # Simulate data
    params = np.r_[phi, theta, Phi, Theta, 1.0]  # Combine all parameters
    params = np.array([i for i in params if i != 0])
    print(params.shape)
    print(params)
    print(f"params: ({p}, {d}, {q}) x ({P}, {D}, {Q})_{s}")
    y = model.simulate(params, n)
    print(y[:10])


    return y

# # Example usage:
# # 2.1
# simulate_seasonal_arima_sarimax(phi=[0.6], d=0, theta=[], Phi=[], D=0, Theta=[], s=12, n=1200, label='Example 2.1')
# # 2.2
# simulate_seasonal_arima_sarimax(phi=[], d=0, theta=[], Phi=[-0.9], D=0, Theta=[], s=12, n=1200, label='Example 2.2')

# # 2.3
# simulate_seasonal_arima_sarimax(phi=[0.9], d=0, theta=[], Phi=[], D=0, Theta=[-0.7], s=12, n=1200, label='Example 2.3')

# # 2.4
# simulate_seasonal_arima_sarimax(phi=[-0.6], d=0, theta=[], Phi=[-0.8], D=0, Theta=[], s=12, n=1200,  label='Example 2.4')

# # 2.5
# simulate_seasonal_arima_sarimax(phi=[], d=0, theta=[0.4], Phi=[], D=0, Theta=[-0.8], s=12, n=1200, label='Example 2.5')

# # 2.6
# simulate_seasonal_arima_sarimax(phi=[], d=0, theta=[-0.4], Phi=[0.7], D=0, Theta=[], s=12, n=1200, label='Example 2.6')

#simulate_seasonal_arima_sarimax(0,0,0,1,0,0,12, 1200)

# Define model parameters
models = [
    {"params": {"phi": 0.6, "theta": 0, "Phi": 0, "Theta": 0}, "p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "s": 12, "title": "(1,0,0)x(0,0,0)"},
    {"params": {"phi": 0, "theta": 0, "Phi": -0.9, "Theta": 0}, "p": 0, "d": 0, "q": 0, "P": 1, "D": 0, "Q": 0, "s": 12, "title": "(0,0,0)x(1,0,0)"},
    {"params": {"phi": 0.9, "theta": -0.7, "Phi": 0, "Theta": 0}, "p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 1, "s": 12, "title": "(1,0,0)x(0,0,1)"},
    {"params": {"phi": -0.6, "theta": 0, "Phi": -0.8, "Theta": 0}, "p": 1, "d": 0, "q": 0, "P": 1, "D": 0, "Q": 0, "s": 12, "title": "(1,0,0)x(1,0,0)"},
    {"params": {"phi": 0, "theta": 0.4, "Phi": 0, "Theta": -0.8}, "p": 0, "d": 0, "q": 1, "P": 0, "D": 0, "Q": 1, "s": 12, "title": "(0,0,1)x(0,0,1)"},
    {"params": {"phi": 0, "theta": -0.4, "Phi": 0.7, "Theta": 0}, "p": 0, "d": 0, "q": 1, "P": 1, "D": 0, "Q": 0, "s": 12, "title": "(0,0,1)x(1,0,0)"}
]

# Simulate and plot each model with ACF and PACF
plt.figure(figsize=(15, 18))
for i, model in enumerate(models):
    # Simulate the time series
    print(f"model 2.{i+1}")
    y = simulate_seasonal_arima_sarimax(
        model["params"]["phi"], model["params"]["theta"],
        model["params"]["Phi"], model["params"]["Theta"],
        model["p"], model["d"], model["q"], model["P"], model["D"], model["Q"], model["s"]
    )
    
    # Plot the time series
    plt.subplot(6, 3, 3*i+1)
    plt.plot(y)
    plt.title(f'{model["title"]} - Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    # Plot the ACF
    plt.subplot(6, 3, 3*i+2)
    plot_acf(y, lags=12*3, ax=plt.gca(), title=f'{model["title"]} - ACF')
    
    # Plot the PACF
    plt.subplot(6, 3, 3*i+3)
    plot_pacf(y, lags=12*3, ax=plt.gca(), title=f'{model["title"]} - PACF')

# Adjust spacing between plots
plt.subplots_adjust(hspace=0.6, wspace=0.4)  # Adjust horizontal and vertical spacing

# If you want a tight layout that fits everything, you can also do:
plt.tight_layout()

plt.savefig('ex2.png')
