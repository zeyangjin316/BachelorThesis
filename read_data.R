# Read the RDS file
data <- readRDS("data_for_kit.rds")

# Convert to regular data frame (removing the grouping)
data_ungrouped <- as.data.frame(data)

# Save to CSV
# Note: The returns_5m column contains lists, which we'll need to handle separately
data_without_lists <- data_ungrouped[, !sapply(data_ungrouped, is.list)]
write.csv(data_without_lists, "data_for_kit.csv", row.names = FALSE)

# Convert the returns_5m data to a matrix and save
returns_5m_df <- do.call(rbind, data$returns_5m)
write.csv(returns_5m_df, "returns_5m.csv", row.names = FALSE)