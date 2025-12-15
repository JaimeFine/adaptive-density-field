library(sf)
library(dplyr)

TIMESTAMP_FORMAT <- "%Y-%m-%d %H:%M:%S"

geojson <- read.csv("shuangliu.csv", stringsAsFactors = FALSE) %>%
  mutate(track_timestamp = as.POSIXct(track_timestamp, format = TIMESTAMP_FORMAT))

geojson_sf <- st_as_sf(
  geojson,
  coords = c("track_longitude", "track_latitude"),
  crs = 4326  # WSG84 coordinate system
)

routes <- geojson_sf %>%
  group_by(identification_number) %>%
  arrange(track_timestamp) %>%
  summarise(geometry = st_combine(geometry)) %>%
  st_cast("LINESTRING")

routes <- routes %>%
  left_join(
    geojson %>%
      select(identification_number, Airline, airport_origin_icao) %>%
      distinct(),
    by = "identification_number"
  )

st_write(routes, "flight_routes.geojson", driver = "GeoJSON", append = FALSE)



