install.packages("glmnet")
library(glmnet)

fit_lasso <- function(y, X) {
  # Convert inputs to matrix format
  X <- as.matrix(X)
  y <- as.vector(y)
  
  # Fit LASSO model with cross-validation to select lambda
  cv_fit <- cv.glmnet(X, y, alpha = 1)
  
  # Fit the final model using the optimal lambda
  final_model <- glmnet(X, y, alpha = 1, lambda = cv_fit$lambda.min)
  
  # Get the coefficients
  coef <- coef(final_model)
  
  # Get fitted values
  fitted_values <- predict(final_model, X)
  
  # Return results
  return(list(
    model = final_model,
    cv_fit = cv_fit,
    coefficients = coef,
    fitted_values = fitted_values,
    lambda_min = cv_fit$lambda.min
  ))
}