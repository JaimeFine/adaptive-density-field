library(dplyr)
library(ggplot2)

df <- read.csv("alpha_sensitivity_results.csv")

summary <- df %>%
  group_by(alpha_val) %>%
  summarise(
    zoi_points = sum(is_zoi == 1),
    communities = n_distinct(community_id[community_id != -1]),
    .groups = "drop"
  )

comm_ticks <- c(1000, 1025, 1050, 1075, 1100)
zoi_ticks  <- c(41000, 40000, 39000, 38000, 37000)

comm_rng <- range(comm_ticks)
zoi_rng  <- range(zoi_ticks)

comm_to_s <- function(y) (y - comm_rng[1]) / (comm_rng[2] - comm_rng[1])
s_to_zoi  <- function(s) s * (zoi_rng[2] - zoi_rng[1]) + zoi_rng[1]

summary <- summary %>%
  mutate(
    comm_s = comm_to_s(communities),
    zoi_s  = (zoi_points - zoi_rng[1]) / (zoi_rng[2] - zoi_rng[1])
  )

ggplot(summary, aes(x = alpha_val)) +
  geom_line(aes(y = comm_s, color = "Communities"), linewidth = 1) +
  geom_point(aes(y = comm_s, color = "Communities")) +
  geom_line(aes(y = zoi_s, color = "ZOI Points"), linewidth = 1, linetype = "dashed") +
  geom_point(aes(y = zoi_s, color = "ZOI Points")) +
  scale_color_manual(
    name = "Legend",
    values = c("Communities" = "blue", "ZOI Points" = "red")
  ) +
  scale_y_continuous(
    name = "Number of Communities",
    limits = c(0, 1),
    breaks = comm_to_s(comm_ticks),
    labels = comm_ticks,
    sec.axis = sec_axis(
      trans = s_to_zoi,
      name = "Number of ZOI Points",
      breaks = zoi_ticks,
      labels = zoi_ticks
    )
  ) +
  labs(x = expression(alpha)) +
  theme_minimal()

summary