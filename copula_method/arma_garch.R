# arma_garch.R
suppressMessages({
  if (!requireNamespace("forecast", quietly = TRUE)) stop("Package 'forecast' missing")
  if (!requireNamespace("rugarch",  quietly = TRUE)) stop("Package 'rugarch'  missing")
})
library(forecast)
library(rugarch)

fit_arma_garch <- function(
  time_series,
  arima_max_p = 5, arima_max_q = 5, seasonal = FALSE,
  ic = "aicc", stepwise = TRUE, approximation = FALSE,
  variance_model = "sGARCH",
  garch_order = c(1, 1),
  mean_order = c(0, 0),
  include_mean = FALSE,
  dist = "norm"
) {
  arma_model <- forecast::auto.arima(
    time_series,
    max.p = arima_max_p,
    max.q = arima_max_q,
    seasonal = seasonal,
    ic = ic,
    stepwise = stepwise,
    approximation = approximation
  )
  arma_residuals <- residuals(arma_model)

  garch_spec <- rugarch::ugarchspec(
    variance.model = list(model = variance_model, garchOrder = garch_order),
    mean.model     = list(armaOrder = mean_order, include.mean = include_mean),
    distribution.model = dist
  )
  garch_model <- rugarch::ugarchfit(spec = garch_spec, data = arma_residuals)

  list(arma_model = arma_model, garch_model = garch_model)
}

forecast_arma_garch_samples <- function(arma_model, garch_model, n_samples = 1000) {
  mu  <- as.numeric(forecast::forecast(arma_model, h = 1)$mean)
  sig <- as.numeric(rugarch::sigma(rugarch::ugarchforecast(garch_model, n.ahead = 1)))
  stats::rnorm(as.integer(n_samples), mean = mu, sd = sig)
}