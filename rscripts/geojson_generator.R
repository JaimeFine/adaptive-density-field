library(dplyr)
library(sf)
library(readr)
library(tools)
library(geosphere)

TIMESTAMP_FORMAT <- "%Y-%m-%d %H:%M:%S"

folder <- "data_folder"
files <- list.files(folder, pattern = "\\.csv$", full.names=TRUE)

for (file in files) {
  
  message("Processing: ", file)

  tryCatch({
    # Basic processing:
    geojson <- read_csv(file, show_col_types=FALSE) %>%
      mutate(
        track_timestamp = as.POSIXct(track_timestamp, format=TIMESTAMP_FORMAT)
      ) %>%
      filter(
        !is.na(track_timestamp),
        !is.na(track_longitude),
        !is.na(track_latitude)
      )
    
    if (nrow(geojson) < 50)
      stop("Not enough valid points")
    
    # Spatialization:
    geojson_sf <- st_as_df(
      geojson,
      coords = c("track_longitude", "track_latitude", "track_altitude"),
      crs = 4326,
      remove = FALSE
    )
    
    # Building trajectories:
    routes <-geojson_sf %>%
      group_by(identification_number) %>%
      arrange(track_timestamp, .by_group=TRUE) %>%
      filter(n() >= 2) %>%
      summarise(
        geometry = st_combine(geometry),
        .groups = "drop"
      ) %>%
      st_cast("LINESTRING")
    
    # Attach metadata:
    meta <- geojsodn %>% %>%
      select(
        identification_number,
        Airline,
        airport_origin_icao,
        identification_id
      ) %>%
      distinct()
    
    routes <- routes %>%
      left_join(meta, by = "identification_number")
    
    # Write the file:
    out_file <- file.path(
      folder,
      paste0(file_path_sans_ext(basename(file)), ".geojson")
    )
    
    st_write(routes, out_file, driver = "GeoJSON", append = FALSE)
    
  }, error = function(e) {
    message("Error in ", file, ": ", e$message)
  })
}

