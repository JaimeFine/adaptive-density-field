library(dplyr)
library(leaflet)
library(scales)
library(sf)

track <- read.csv("trajectory_adf_zoi.csv")

track_filtered <- track[track$ADF <= 15,]

m <- leaflet() %>%
  addProviderTiles("CartoDB.DarkMatter") %>%
  
  # ZOI points (optional, if needed)
  addCircleMarkers(
    data = filter(track_filtered, ZOI == 1),
    lng = ~lon,
    lat = ~lat,
    radius = 1,
    color = "red",
    fillOpacity = 1.0,
    group = "ZOI-points"
  )

m
