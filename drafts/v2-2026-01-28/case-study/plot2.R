library(dplyr)
library(leaflet)
library(scales)
library(sf)

poi <- read.csv("poi_background.csv")
track <- read.csv("trajectory_adf_zoi.csv")
chengdu <- st_read("chengdu.geo.json")
zoi_sf <- st_read("zoi_polygons_meters_mid.geojson")

zoi_sf <- st_set_crs(zoi_sf, 4326)
if (st_crs(zoi_sf)$epsg != 4326) zoi_sf <- st_transform(zoi_sf, 4326)

if (!"community" %in% names(zoi_sf)) zoi_sf$community <- seq_len(nrow(zoi_sf))

poi <- st_as_sf(poi, coords = c("lon", "lat"), crs = 4326, remove = FALSE)
bbox <- st_bbox(chengdu)
poi_chengdu <- poi %>%
  filter(
    st_coordinates(.)[,1] >= bbox["xmin"] &
      st_coordinates(.)[,1] <= bbox["xmax"] &
      st_coordinates(.)[,2] >= bbox["ymin"] &
      st_coordinates(.)[,2] <= bbox["ymax"]
  )

track_filtered <- track[track$ADF <= 15,]
pal_adf <- colorNumeric(palette = "viridis", domain = track_filtered$ADF)

zoi_sf <- st_make_valid(zoi_sf)
poly_idx <- st_geometry_type(zoi_sf) %in% c("POLYGON", "MULTIPOLYGON")
zoi_sf <- zoi_sf[poly_idx, ]
zoi_sf <- st_cast(zoi_sf, "MULTIPOLYGON", warn = FALSE)  # ensure uniform type

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
  
  # ZOIs as polygons
  addPolygons(
    data = zoi_sf,
    color = "red",
    weight = 1,
    opacity = 0.8,
    fillOpacity = 0.5,
    group = "ZOI"
  ) %>%
  
  # ZOI points (optional, if needed)
  addCircleMarkers(
    data = filter(track_filtered, ZOI == 1),
    lng = ~lon,
    lat = ~lat,
    radius = 1,
    color = "red",
    fillOpacity = 1.0,
    group = "ZOI-points"
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
    data = track_filtered,
    lng = ~lon,
    lat = ~lat,
    color = "white",
    weight = 2,
    opacity = 0.8,
    group = "Trajectory"
  ) %>%
  
  # Legend
  addLegend(
    position = "bottomright",
    pal = pal_adf,
    values = track_filtered$ADF,
    title = "ADF Value",
    opacity = 1
  ) %>%
  
  # Layers control
  addLayersControl(
    overlayGroups = c("POIs", "ADF", "ZOI", "ZOI-points", "Trajectory"),
    options = layersControlOptions(collapsed = FALSE)
  )

m
