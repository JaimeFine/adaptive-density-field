library(dplyr)
library(leaflet)

poi_processed <- read.csv("data/2024-12-16-CTU_processed_poi.csv")
adf_processed <- read.csv("data/filtered_poi.csv")

view <- leaflet() %>%
  addProviderTiles("CartoDB.DarkMatter") %>%
  setView(lng = 104.0638, lat = 30.5754, zoom = 9) %>%
  # Physical Baseline
  addCircleMarkers(
    data = poi_processed,
    lng = ~lon, lat = ~lat,
    radius = 1, color = "#FF4B2B",
    stroke = FALSE, fillOpacity = 0.8,
    group = "Kinematic Baseline"
  ) %>%
  # ADF ZOI Points
  addCircleMarkers(
    data = adf_processed,
    lng = ~lon, lat = ~lat,
    radius = 1, color = "#00E5FF",
    stroke = FALSE, fillOpacity = 0.4,
    group = "ADF Framework"
  ) %>%
  addLayersControl(
    overlayGroups = c("Kinematic Baseline", "ADF Framework"),
    options = layersControlOptions(collapsed = FALSE)
  )

view

