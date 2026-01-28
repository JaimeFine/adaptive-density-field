# This program is to visualize the Mahalanobis losses' distributions

library(dplyr)

loss <- read.csv("flight_loss.csv")

loss1 <- loss %>%
  filter(flight_id == "3U3052")

loss2 <- loss %>%
  filter(flight_id == "CZ6773")

loss3 <- loss %>%
  filter(flight_id == "OQ2104")

par(mfrow = c(2, 2))

hist(
  loss1$mahalanobis_loss,
  main = "Loss",
  breaks = 20,
  xlab = "Mahalanobis Loss",
  xlim = c(0, 5),
  col = "darkorange"
)

hist(
  loss2$mahalanobis_loss,
  main = "Loss",
  breaks = 20,
  xlab = "Mahalanobis Loss",
  xlim = c(0, 5),
  col = "darkred"
)

hist(
  loss3$mahalanobis_loss,
  main = "Loss",
  breaks = 20,
  xlab = "Mahalanobis Loss",
  xlim = c(0, 5),
  col = "darkcyan"
)

hist(
  loss$mahalanobis_loss,
  main = "Loss",
  breaks = 100,
  xlab = "Mahalanobis Loss",
  xlim = c(0, 5),
  col = "darkgreen"
)


