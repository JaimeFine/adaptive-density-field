library(ggplot2)

df <- read.csv("data/distance_sensitivity.csv")

ggplot(df, aes(x = Threshold_m)) +
  geom_line(
    aes(
      y = Precision * 100,
      color = "Precision"
    ),
    linewidth = 1
  ) +
  geom_line(
    aes(
      y = Recall * 100,
      color = "Recall"
    ),
    linewidth = 1
  ) +
  geom_line(
    aes(
      y = F1 * 100,
      color = "F1"
    ),
    linewidth = 1
  ) +
  scale_color_manual(
    name = "Metric",
    values = c("Precision" = "blue", "Recall" = "red", "F1" = "orange")
  ) +
  theme_minimal() +
  labs(
    x = "Distance Threshold (meters)",
    y = "Metrics (%)"
  ) +
  scale_x_continuous(breaks = seq(100, 500, 50)) +
  ylim(0, 100)
