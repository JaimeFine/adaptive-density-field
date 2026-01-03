library(dplyr)
library(leaflet)
library(scales)

poi <- read.csv("poi_background.csv")
traj <- read.csv("trajectory_adf_zoi.csv")

set.seed(42)
poi_vis <- poi %>% sample_n(min(100000, n()))

pal_adf <- colorNumeric(
  palette = "viridis",
  domain = traj$ADF
)

m <- leaflet() %>%
  addProviderTiles("CartoDB.DarkMatter")

m <- m %>%
  addCircleMarkers(
    data = poi_vis,
    lng = ~lon,
    lat = ~lat,
    radius = 1,
    color = "gray",
    opacity = 0.15,
    fillOpacity = 0.15,
    group = "POIs"
  )

m <- m %>%
  addCircleMarkers(
    data = filter(traj, ZOI == 1),
    lng = ~lon,
    lat = ~lat,
    radius = 1,
    color = "red",
    fillOpacity = 1.0,
    group = "ZOI"
  )

m <- m %>%
  addCircleMarkers(
    data = traj,
    lng = ~lon,
    lat = ~lat,
    radius = 3,
    color = ~pal_adf(ADF),
    stroke = FALSE,
    fillOpacity = 0.8,
    group = "ADF"
  )


m <- m %>%
  addPolylines(
    lng = traj$lon,
    lat = traj$lat,
    color = "white",
    weight = 2,
    opacity = 0.8,
    group = "Trajectory"
  )

m <- m %>%
  addLegend(
    position = "bottomright",
    pal = pal_adf,
    values = traj$ADF,
    title = "ADF Value",
    opacity = 1
  ) %>%
  addLayersControl(
    overlayGroups = c("POIs", "ADF", "ZOI", "Trajectory"),
    options = layersControlOptions(collapsed = FALSE)
  )

m
