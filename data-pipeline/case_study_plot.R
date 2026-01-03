library(dplyr)
library(leaflet)
library(scales)
library(sf)

poi <- read.csv("poi_background.csv")
track <- read.csv("trajectory_adf_zoi.csv")
chengdu <- st_read("chengdu.geo.json")

poi <- st_as_sf(
  poi, coords = c("lon", "lat"), crs = 4326
)

poi_chengdu <- st_intersection(poi, chengdu)

pal_adf <- colorNumeric(
  palette = "viridis",
  domain = track$ADF
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
    data = filter(track, ZOI == 1),
    lng = ~lon,
    lat = ~lat,
    radius = 1,
    color = "red",
    fillOpacity = 1.0,
    group = "ZOI"
  )

m <- m %>%
  addCircleMarkers(
    data = track,
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
    lng = track$lon,
    lat = track$lat,
    color = "white",
    weight = 2,
    opacity = 0.8,
    group = "Trajectory"
  )

m <- m %>%
  addLegend(
    position = "bottomright",
    pal = pal_adf,
    values = track$ADF,
    title = "ADF Value",
    opacity = 1
  ) %>%
  addLayersControl(
    overlayGroups = c("POIs", "ADF", "ZOI", "Trajectory"),
    options = layersControlOptions(collapsed = FALSE)
  )

m
