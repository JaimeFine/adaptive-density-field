library(sf)
library(ggplot2)
library(dplyr)
library(leaflet)
library(leaflet.extras)

pois <- read.csv("poi_data.csv")

set.seed(42)
pois_vis <- pois %>% sample_n(min(30000, n()))

leaflet(pois_vis) %>%
  addProviderTiles("CartoDB.Positron") %>%
  setView(lng = 104.0667, lat = 30.6667, zoom = 3) %>%
  addHeatmap(
    lng = ~lon,
    lat = ~lat,
    intensity = ~poi_score,
    blur = 10,
    max = max(pois_vis$poi_score, na.rm = TRUE)
  )