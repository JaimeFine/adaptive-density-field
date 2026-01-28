library(ggplot2)
library(dplyr)

data <- read.csv("sigma_sensitivity_report_meters.csv")

ggplot(data, aes(x = sigma, y = num_communities)) +
  geom_line(color = "darkgreen", linewidth = 1) +
  geom_point(color = "darkgreen", size = 2) +
  scale_y_continuous(
    limits = c(400, 1600),   # descending axis
    breaks = c(400, 800, 1200, 1600),
    name = "Number of ZOI Points"
  ) +
  labs(
    x = expression(sigma),
    y = "Number of Communities"
  ) +
  theme_minimal()

data