library(sf)
library(dplyr)
library(units)

df_phys <- read.csv("data/2024-12-16-CTU_processed_poi.csv")
df_adf  <- read.csv("data/filtered_poi.csv")

p_phys <- st_as_sf(df_phys, coords = c("lon", "lat", "alt"), crs = 4326)
p_adf  <- st_as_sf(df_adf, coords = c("lon", "lat", "alt"), crs = 4326)

p_phys <- st_transform(p_phys, 4978)
p_adf <- st_transform(p_adf, 4978)

thresholds_m <- c(100, 150, 200, 300, 400, 500)

results <- data.frame()

for (th in thresholds_m) {
  threshold <- set_units(th, "m")
  
  # Physical to ADF
  dist_p_to_a <- st_is_within_distance(p_phys, p_adf, dist = threshold)
  df_phys$type <- ifelse(lengths(dist_p_to_a) > 0, "TT", "TF")
  
  # ADF to Physical
  dist_a_to_p <- st_is_within_distance(p_adf, p_phys, dist = threshold)
  df_adf$type <- ifelse(lengths(dist_a_to_p) > 0, "TT", "FT")
  
  tt <- sum(df_adf$type == "TT")
  fp <- sum(df_adf$type == "FT")
  fn <- sum(df_phys$type == "TF")
  
  precision <- if (tt + fp > 0) tt / (tt + fp) else 0
  recall <- if (tt + fn > 0) tt / (tt + fn) else 0
  f1 <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
  
  results <- rbind(results, data.frame(
    Threshold_m = th,
    TT = tt,
    FP = fp,
    FN = fn,
    Precision = precision,
    Recall = recall,
    F1 = f1
  ))
}

print(results)
write.csv(results, "distance_sensitivity.csv", row.names = FALSE)


