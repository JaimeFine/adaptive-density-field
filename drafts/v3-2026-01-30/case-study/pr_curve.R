library(sf)
library(dplyr)
library(ggplot2)

# Load data
df_phys <- read.csv("data/2024-12-16-CTU_processed_poi.csv")
df_adf  <- read.csv("data/filtered_poi.csv")

p_phys <- st_as_sf(df_phys, coords = c("lon", "lat"), crs = 4326)
p_adf  <- st_as_sf(df_adf,  coords = c("lon", "lat"), crs = 4326)

p_phys <- st_transform(p_phys, 3857)
p_adf <- st_transform(p_adf, 3857)

# --- PARAMETERS ---
dist_thresh <- 200
score_thresholds <- quantile(df_adf$ADF, probs = seq(0.05, 0.95, 0.05))

# --- PR COMPUTATION ---
pr_results <- lapply(score_thresholds, function(th) {
  
  preds <- p_adf[df_adf$ADF >= th, ]
  
  if (nrow(preds) == 0) {
    return(data.frame(
      threshold = th,
      precision = NA,
      recall = 0
    ))
  }
  
  # Match predicted → ground truth
  match_pred <- st_is_within_distance(preds, p_phys, dist = dist_thresh)
  TP <- sum(lengths(match_pred) > 0)
  FP <- nrow(preds) - TP
  
  # Match ground truth → predicted
  match_gt <- st_is_within_distance(p_phys, preds, dist = dist_thresh)
  FN <- sum(lengths(match_gt) == 0)
  
  precision <- TP / (TP + FP)
  recall    <- TP / (TP + FN)
  
  data.frame(
    threshold = th,
    precision = precision,
    recall = recall
  )
})

pr_df <- do.call(rbind, pr_results) %>% na.omit()

ggplot(pr_df, aes(x = recall, y = precision)) +
  geom_line(linewidth = 1, color = "steelblue") +
  geom_point(size = 2, color = "steelblue") +
  theme_minimal() +
  labs(
    title = "Precision–Recall Curve for POI Detection",
    x = "Recall",
    y = "Precision"
  ) +
  scale_x_continuous(breaks = seq(0, 1, 0.25)) +
  ylim(0, 1)
