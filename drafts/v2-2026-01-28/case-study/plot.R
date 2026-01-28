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

bbox <- st_bbox(chengdu)
poi_chengdu <- poi %>%
  filter(
    st_coordinates(.)[,1] >= bbox["xmin"] &
      st_coordinates(.)[,1] <= bbox["xmax"] &
      st_coordinates(.)[,2] >= bbox["ymin"] &
      st_coordinates(.)[,2] <= bbox["ymax"]
  )

track_filtered <- track[track$ADF <= 15,]

pal_adf <- colorNumeric(
  palette = "viridis",
  domain = track_filtered$ADF
)
track
m <- leaflet() %>%
  addProviderTiles("CartoDB.DarkMatter") %>%
  # POIs
  addCircleMarkers(
    data = poi_chengdu,
    lng = ~st_coordinates(poi_chengdu)[,1],
    lat = ~st_coordinates(poi_chengdu)[,2],
    radius = 1,
    color = "gray",
    opacity = 0.15,
    fillOpacity = 0.15,
    group = "POIs"
  ) %>%
  # ZOIs
  addCircleMarkers(
    data = filter(track_filtered, ZOI == 1),
    lng = ~lon,
    lat = ~lat,
    radius = 1,
    color = "red",
    fillOpacity = 1.0,
    group = "ZOI"
  ) %>%
  # ADF Field
  addCircleMarkers(
    data = track,
    lng = ~lon,
    lat = ~lat,
    radius = 1,
    color = ~pal_adf(ADF),
    stroke = FALSE,
    fillOpacity = 0.8,
    group = "ADF"
  ) %>%
  # Trajectory
  addPolylines(
    lng = track_filtered$lon,
    lat = track_filtered$lat,
    color = "white",
    weight = 2,
    opacity = 0.8,
    group = "Trajectory"
  ) %>%
  addLegend(
    position = "bottomright",
    pal = pal_adf,
    values = track_filtered$ADF,
    title = "ADF Value",
    opacity = 1
  ) %>%
  addLayersControl(
    overlayGroups = c("POIs", "ADF", "ZOI", "Trajectory"),
    options = layersControlOptions(collapsed = FALSE)
  )

m
