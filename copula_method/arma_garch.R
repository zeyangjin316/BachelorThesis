install.packages("forecast")
install.packages("rugarch")

library(forecast)
library(rugarch)

fit_arma_garch <- function(time_series) {
  # Fit ARMA model
  arma_model <- auto.arima(time_series)
  arma_residuals <- residuals(arma_model)

  # Specify the GARCH model (here we use a basic GARCH(1,1) model)
  garch_spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                           mean.model = list(armaOrder = c(0, 0)),
                           distribution.model = "norm")

  # Fit the GARCH model to the residuals of the ARMA model
  garch_model <- ugarchfit(spec = garch_spec, data = arma_residuals)

  # Return the results of the ARMA and GARCH models
  return(list(arma_model = arma_model, garch_model = garch_model))
}

forecast_arma_garch_samples <- function(arma_model, garch_model, n_samples = 1000) {
  # Forecast mean from ARMA
  mu <- as.numeric(forecast(arma_model, h = 1)$mean)

  # Forecast volatility from GARCH
  sigma <- sigma(ugarchforecast(garch_model, n.ahead = 1))

  # Generate samples from predictive distribution
  samples <- rnorm(n_samples, mean = mu, sd = sigma)
  return(samples)
}