library(dplyr)
library(leaflet)
library(scales)
library(sf)

track <- read.csv("trajectory_adf_zoi.csv")

m <- leaflet() %>%
  addProviderTiles("CartoDB.DarkMatter") %>%
  
  # ZOI points (optional, if needed)
  addCircleMarkers(
    data = filter(track, ZOI == 1),
    lng = ~lon,
    lat = ~lat,
    radius = 1,
    color = "red",
    fillOpacity = 1.0,
    group = "ZOI-points"
  )

m
