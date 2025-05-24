kf_predict <- function(params, df) {
  A   <- matrix(params[1], nrow = 1, ncol = 1)
  B   <- matrix(params[2:4], nrow = 1)
  Sigma1lt <- matrix(params[5], nrow = 1, ncol = 1)
  Sigma1   <- Sigma1lt %*% t(Sigma1lt)
  C   <- matrix(params[6], nrow = 1, ncol = 1)
  Sigma2 <- matrix(params[7]^2, nrow = 1, ncol = 1)
  X0  <- matrix(20, ncol = 1)
  
  Y <- as.matrix(df[, "Y"])
  U <- as.matrix(df[, c("Ta", "S", "I")])
  Tn <- nrow(df)
  
  n <- nrow(A)
  x_est <- X0
  P_est <- diag(1e1, n)
  
  y_preds <- numeric(Tn)
  
  for (t in 1:Tn) {
    # Prediction
    x_pred <- A %*% x_est + B %*% matrix(U[t, ], ncol = 1)
    P_pred <- A %*% P_est %*% t(A) + Sigma1
    
    y_pred <- C %*% x_pred
    y_preds[t] <- y_pred
    
    # Innovation and update
    S_t <- C %*% P_pred %*% t(C) + Sigma2
    innov <- Y[t, ] - y_pred
    K_t <- P_pred %*% t(C) %*% solve(S_t)
    x_est <- x_pred + K_t %*% innov
    P_est <- (diag(n) - K_t %*% C) %*% P_pred
  }
  
  return(data.frame(Time = 1:Tn, Observed = Y[,1], Predicted = y_preds))
}


kf_with_residuals <- function(par, df) {
  A   <- matrix(par[1], nrow = 1, ncol = 1)
  B   <- matrix(par[2:4], nrow = 1)
  Sigma1lt <- matrix(par[5], nrow = 1, ncol = 1)
  Sigma1   <- Sigma1lt %*% t(Sigma1lt)
  C   <- matrix(par[6], nrow = 1, ncol = 1)
  Sigma2 <- matrix(par[7]^2, nrow = 1, ncol = 1)
  X0  <- matrix(20, ncol = 1)
  
  obs_cols <- c("Y")
  input_cols <- c("Ta","S","I")
  Y  <- as.matrix(df[, obs_cols])
  U  <- as.matrix(df[, input_cols])
  Tn <- nrow(df)
  
  n      <- nrow(A)
  x_est  <- X0
  P_est  <- diag(1e1, n)
  residuals <- numeric(Tn)
  
  for (t in 1:Tn) {
    x_pred <- A %*% x_est + B %*% matrix(U[t, ], ncol = 1)
    P_pred <- A %*% P_est %*% t(A) + Sigma1
    
    y_pred  <- C %*% x_pred
    S_t     <- C %*% P_pred %*% t(C) + Sigma2
    innov   <- matrix(Y[t, ], ncol = 1) - y_pred
    
    residuals[t] <- innov
    
    K_t    <- P_pred %*% t(C) %*% solve(S_t)
    x_est  <- x_pred + K_t %*% innov
    P_est  <- (diag(n) - K_t %*% C) %*% P_pred
  }
  
  residuals
}


kf_logLik_dt <- function(par, df) {
  # par: vector of parameters
  # df: data frame with observations and inputs as columns (Y, Ta, S, I)
  # par: Could be on the form c(A11, A12, A21, A22, B11, B12, B21, B22, Q11, Q12, Q22)
  A   <- matrix(par[1], nrow = 1, ncol = 1) # transition matrix
  B   <- matrix(par[2:4], nrow = 1) # input matrix
  Sigma1lt <- matrix(par[5], nrow = 1, ncol = 1) # lower-triangle of system covariance matrix
  Sigma1   <- Sigma1lt %*% t(Sigma1lt) # THAT IS!!! The system covariance matrix is given by Qlt %*% t(Qlt) (and is this symmetric positive definite)
  C   <- matrix(par[6], nrow = 1, ncol = 1) # observation matrix
  Sigma2 <- matrix(par[7]^2, nrow = 1, ncol = 1) # observation noise covariance matrix
  X0  <- matrix(20, ncol = 1) # initial state

  # Variables
  obs_cols <- c("Y") # observation column names
  input_cols <- c("Ta","S","I") # input column names

  # pull out data
  Y  <- as.matrix(df[, obs_cols])     # m×T
  U  <- as.matrix(df[, input_cols])   # p×T
  Tn <- nrow(df)

  # init
  n      <- nrow(A)
  x_est  <- matrix(Y[1,], n, 1)            # start state from first obs
  x_est <- X0 
  P_est  <- diag(1e1, n)                   # X0 prior covariance
  logLik <- 0

  for (t in 1:Tn) {
    # prediction step
    x_pred <- A %*% x_est + B %*% matrix(U[t, ], ncol = 1)# write the prediction step
    P_pred <- A %*% P_est %*% t(A) + Sigma1 # write the prediction step (Sigma_xx)

    # innovation step
    y_pred  <- C %*% x_pred# predicted observation
    S_t     <- C %*% P_pred %*% t(C) + Sigma2 # predicted observation covariance (Sigma_yy)
    innov   <- matrix(Y[t, ], ncol = 1) - y_pred # innovation (one-step prediction error)

    # log-likelihood contribution
    logLik <- logLik - 0.5*(sum(log(2*pi*S_t)) + t(innov) %*% solve(S_t, innov))

    # update step
    K_t    <- P_pred %*% t(C) %*% solve(S_t) # Kalman gain
    x_est  <- x_pred + K_t %*% innov # reconstructed state
    P_est  <- (diag(n) - K_t %*% C) %*% P_pred# reconstructed covariance
  }

  as.numeric(logLik)
}

# Optimizer wrapper
estimate_dt <- function(start_par, df, lower=NULL, upper=NULL) {
  negLL <- function(par){ -kf_logLik_dt(par, df) }
  optim(
    par    = start_par, fn = negLL,
    method = "L-BFGS-B",
    lower  = lower, upper = upper,
    control= list(maxit=1000, trace=1)
  )
}

start_par <- c(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
lower <- c(-5,-5,-5,-5,-5,-5,-5)
upper <- c(5,5,5,5,5,5,5)
library(readr)
transformer_data <- read_csv("C:/Timeseries/assignment4/transformer_data.csv")
transformer_data <- transformer_data[,-1]


result <- estimate_dt(start_par, transformer_data, lower = lower , upper = upper)

par_out <- result$par


library(ggplot2)
library(forecast)
library(gridExtra)

# Get residuals
residuals <- kf_with_residuals(par_out, transformer_data)

# Residual Plot
p1 <- ggplot(data.frame(Time = 1:length(residuals), Residuals = residuals), aes(x = Time, y = Residuals)) +
  geom_line() + ggtitle("Residual Plot") + theme_minimal()

# QQ Plot
p2 <- ggplot(data.frame(res = residuals), aes(sample = res)) +
  stat_qq() + stat_qq_line() + ggtitle("QQ-Plot of Residuals") + theme_minimal()

# ACF
p3 <- ggAcf(residuals, main = "ACF of Residuals")

# PACF
p4 <- ggPacf(residuals, main = "PACF of Residuals")

# Show all plots
grid.arrange(p1, p2, p3, p4, nrow = 2)

prediction_df <- kf_predict(par_out, transformer_data)


ggplot(prediction_df, aes(x = Time)) +
  geom_line(aes(y = Observed), color = "black", size = 1, linetype = "solid") +
  geom_line(aes(y = Predicted), color = "blue", size = 1, linetype = "dashed") +
  labs(
    title = "Observed vs Predicted Values - 1D model",
    y = "Value", x = "Time"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 14),
    legend.position = "none"
  )

