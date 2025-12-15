library(leaflet)
library(ggplot2)
library(dplyr)
library(sf)
library(dbscan)
library(leaflet.extras)

# All airport_destination_icao = ZUUU
# All time_estimated_arrival = NA
# All airport_destination_longitude = 103.947
# All airport_destination_latitude = 30.57852
# All airport_destination_offset = 28800
# All time_estimated_departure = ""

# Test graph:

flights <- read.csv("cleaned.csv")
chengdu <- st_read("chengdu.geo.json")

mapping <- ggplot(
  flights, aes(x = airport_origin_longitude, y = airport_origin_latitude)
  ) +
  geom_point() +
  geom_point(aes(x = 103.947, y = 30.57852), color = "red")

# Note that the identification_id + number is for the identification

origin_unique <- unique(
  flights[, c("airport_origin_longitude", "airport_origin_latitude")]
)

track_unique <- unique(
  flights[, c("track_longitude", "track_latitude")]
)

leaflet() %>%
  addTiles() %>%
  setView(lng = 104.0667, lat = 30.6667, zoom = 10) %>%
  addPolygons(
    data = chengdu,
    color = "red",
    fillColor = "orange",
    fillOpacity = 0.4,
    weight = 2,
    popup = "Chengdu City Boundary"
  ) %>%
  addCircleMarkers(
    lng = 103.947, lat = 30.57852,
    color = "blue", radius = 1
  ) %>%
  addCircleMarkers(
    data = origin_unique,
    lng = ~airport_origin_longitude,
    lat = ~airport_origin_latitude,
    color = "red", radius = 1
  ) %>%
  addPolylines(
    data = track_unique,
    lng = ~track_longitude,
    lat = ~track_latitude,
    color = "green", weight = 1,
    opacity = 0.5
  )

# main graph:

flights_sf <- st_as_sf(
  flights, coords = c("track_longitude", "track_latitude"), crs = 4326
)

flights_chengdu <- st_intersection(flights_sf, chengdu)
coords <- st_coordinates(flights_chengdu)

flights_chengdu <- st_filter(flights_sf, chengdu)

leaflet() %>%
  addTiles() %>%
  setView(lng = 104.0667, lat = 30.6667, zoom = 9) %>%
  addHeatmap(
    lng = coords[, 1],
    lat = coords[, 2],
    intensity = 1,
    blur = 35,
    max = 0.005
  ) %>%
  addPolygons(
    data = chengdu,
    color = "red",
    fillColor = "orange",
    fillOpacity = 0.4,
    weight = 2
  )

# Using DBSCAN:
db <- dbscan(coords[, 1:2], eps=0.01, minPts=50)
flights_chengdu$cluster <- db$cluster

leaflet(flights_chengdu) %>%
  addTiles() %>%
  setView(lng = 104.0667, lat = 30.6667, zoom = 9) %>%
  addCircleMarkers(
    lng = ~coords[, 1],
    lat = ~coords[, 2],
    color = ~factor(cluster),
    radius = 2,
    opacity = 1,
    fillOpacity = 0.7
  ) %>%
  addPolygons(
    data = chengdu,
    weight = 1
  )






