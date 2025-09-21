# run this from the project root
library(readr)
library(dplyr)
library(tidyr)

# 1) read the RDS from the data/ folder
data <- readRDS("data/data_for_kit.rds") |> as.data.frame()

# 2) write the base CSV (no list columns) to root
base <- data[, !sapply(data, is.list)]
write_csv(base, "data_for_kit.csv")

# 3) expand the 5m list-column to wide columns while keeping keys
intraday <- data |>
  select(date, sym_root, permno, returns_5m) |>
  tidyr::unnest_wider(returns_5m, names_sep = "_") |>
  # unnest_wider will make columns like returns_5m_1, returns_5m_2, ...
  dplyr::rename_with(
    ~ paste0("V", seq_along(.)),
    .cols = dplyr::starts_with("returns_5m_")
  )

# write intraday CSV to root
write_csv(intraday, "returns_5m.csv")