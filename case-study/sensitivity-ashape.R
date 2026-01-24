library(dplyr)
library(leaflet)
library(scales)
library(sf)

zoi_sf <- st_read("zoi_polygons_meters_low.geojson")

zoi_sf <- st_set_crs(zoi_sf, 4326)
if (st_crs(zoi_sf)$epsg != 4326) zoi_sf <- st_transform(zoi_sf, 4326)

if (!"community" %in% names(zoi_sf)) zoi_sf$community <- seq_len(nrow(zoi_sf))

zoi_sf <- st_make_valid(zoi_sf)
poly_idx <- st_geometry_type(zoi_sf) %in% c("POLYGON", "MULTIPOLYGON")
zoi_sf <- zoi_sf[poly_idx, ]
zoi_sf <- st_cast(zoi_sf, "MULTIPOLYGON", warn = FALSE)  # ensure uniform type

m <- leaflet() %>%
  addProviderTiles("CartoDB.DarkMatter") %>%
  
  # ZOIs as polygons
  addPolygons(
    data = zoi_sf,
    color = "red",
    weight = 1,
    opacity = 0.8,
    fillOpacity = 0.5,
    group = "ZOI"
  )

m
