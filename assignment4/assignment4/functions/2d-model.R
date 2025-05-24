# Required libraries
library(ggplot2)
library(forecast)
library(gridExtra)
library(readr)
transformer_data <- read_csv("C:/Timeseries/assignment4/transformer_data.csv")
transformer_data <- transformer_data[,-1]

# Kalman Filter Prediction Function (2D state, 1D observation)
kf_predict <- function(params, df) {
  A <- matrix(params[1:4], nrow = 2, byrow = TRUE)
  B <- matrix(params[5:10], nrow = 2, byrow = TRUE)
  Sigma1lt <- matrix(c(params[11], 0, params[12], params[13]), nrow = 2)
  Sigma1 <- Sigma1lt %*% t(Sigma1lt)
  C <- matrix(params[14:15], nrow = 1)
  Sigma2 <- matrix(params[16]^2, nrow = 1, ncol = 1)
  X0 <- matrix(c(20, 20), ncol = 1)
  
  Y <- as.matrix(df[, "Y"])
  U <- as.matrix(df[, c("Ta", "S", "I")])
  Tn <- nrow(df)
  
  x_est <- X0
  P_est <- diag(1e1, 2)
  y_preds <- numeric(Tn)
  
  for (t in 1:Tn) {
    x_pred <- A %*% x_est + B %*% matrix(U[t, ], ncol = 1)
    P_pred <- A %*% P_est %*% t(A) + Sigma1
    
    y_pred <- C %*% x_pred
    y_preds[t] <- y_pred
    
    S_t <- C %*% P_pred %*% t(C) + Sigma2
    innov <- Y[t, ] - y_pred
    K_t <- P_pred %*% t(C) %*% solve(S_t)
    x_est <- x_pred + K_t %*% innov
    P_est <- (diag(2) - K_t %*% C) %*% P_pred
  }
  
  return(data.frame(Time = 1:Tn, Observed = Y[,1], Predicted = y_preds))
}

# Residual calculation function
kf_with_residuals <- function(par, df) {
  A <- matrix(par[1:4], nrow = 2, byrow = TRUE)
  B <- matrix(par[5:10], nrow = 2, byrow = TRUE)
  Sigma1lt <- matrix(c(par[11], 0, par[12], par[13]), nrow = 2)
  Sigma1 <- Sigma1lt %*% t(Sigma1lt)
  C <- matrix(par[14:15], nrow = 1)
  Sigma2 <- matrix(par[16]^2, nrow = 1)
  X0 <- matrix(c(20, 20), ncol = 1)
  
  Y <- as.matrix(df[, "Y"])
  U <- as.matrix(df[, c("Ta", "S", "I")])
  Tn <- nrow(df)
  
  x_est <- X0
  P_est <- diag(1e1, 2)
  residuals <- numeric(Tn)
  
  for (t in 1:Tn) {
    x_pred <- A %*% x_est + B %*% matrix(U[t, ], ncol = 1)
    P_pred <- A %*% P_est %*% t(A) + Sigma1
    
    y_pred <- C %*% x_pred
    S_t <- C %*% P_pred %*% t(C) + Sigma2
    innov <- Y[t, ] - y_pred
    
    residuals[t] <- innov
    K_t <- P_pred %*% t(C) %*% solve(S_t)
    x_est <- x_pred + K_t %*% innov
    P_est <- (diag(2) - K_t %*% C) %*% P_pred
  }
  
  residuals
}

# Log-Likelihood function
kf_logLik_dt <- function(par, df) {
  A <- matrix(par[1:4], nrow = 2, byrow = TRUE)
  B <- matrix(par[5:10], nrow = 2, byrow = TRUE)
  Sigma1lt <- matrix(c(par[11], 0, par[12], par[13]), nrow = 2)
  Sigma1 <- Sigma1lt %*% t(Sigma1lt)
  C <- matrix(par[14:15], nrow = 1)
  Sigma2 <- matrix(par[16]^2, nrow = 1)
  X0 <- matrix(c(20, 20), ncol = 1)
  
  Y <- as.matrix(df[, "Y"])
  U <- as.matrix(df[, c("Ta", "S", "I")])
  Tn <- nrow(df)
  
  x_est <- X0
  P_est <- diag(1e1, 2)
  logLik <- 0
  
  for (t in 1:Tn) {
    x_pred <- A %*% x_est + B %*% matrix(U[t, ], ncol = 1)
    P_pred <- A %*% P_est %*% t(A) + Sigma1
    
    y_pred <- C %*% x_pred
    S_t <- C %*% P_pred %*% t(C) + Sigma2
    innov <- Y[t, ] - y_pred
    
    logLik <- logLik - 0.5 * (log(2 * pi * S_t) + (innov^2) / S_t)
    
    K_t <- P_pred %*% t(C) %*% solve(S_t)
    x_est <- x_pred + K_t %*% innov
    P_est <- (diag(2) - K_t %*% C) %*% P_pred
  }
  
  as.numeric(logLik)
}

# Optimization wrapper
estimate_dt <- function(start_par, df, lower=NULL, upper=NULL) {
  negLL <- function(par) { -kf_logLik_dt(par, df) }
  optim(
    par    = start_par, fn = negLL,
    method = "L-BFGS-B",
    lower  = lower, upper = upper,
    control= list(maxit=1000, trace=1)
  )
}

# Starting parameters (4 for A, 6 for B, 3 for Sigma1lt, 2 for C, 1 for Sigma2)
start_par <- c(rep(0.1, 4 + 6 + 3 + 2 + 1))
lower     <- rep(-5, 16)
upper     <- rep(5, 16)

# Run estimation
result <- estimate_dt(start_par, transformer_data, lower = lower, upper = upper)
par_out <- result$par

# Residual analysis
residuals <- kf_with_residuals(par_out, transformer_data)

p1 <- ggplot(data.frame(Time = 1:length(residuals), Residuals = residuals), aes(x = Time, y = Residuals)) +
  geom_line() + ggtitle("Residual Plot") + theme_minimal()

p2 <- ggplot(data.frame(res = residuals), aes(sample = res)) +
  stat_qq() + stat_qq_line() + ggtitle("QQ-Plot of Residuals") + theme_minimal()

p3 <- ggAcf(residuals, main = "ACF of Residuals")
p4 <- ggPacf(residuals, main = "PACF of Residuals")

grid.arrange(p1, p2, p3, p4, nrow = 2)

# Plot predictions
prediction_df <- kf_predict(par_out, transformer_data)

ggplot(prediction_df, aes(x = Time)) +
  geom_line(aes(y = Observed), color = "black", size = 1, linetype = "solid") +
  geom_line(aes(y = Predicted), color = "blue", size = 1, linetype = "dashed") +
  labs(
    title = "Observed vs Predicted Values - 2D model",
    y = "Value", x = "Time"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 14),
    legend.position = "none"
  )

print(par_out)
