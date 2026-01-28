library(dplyr)
library(sf)
library(readr)
library(tools)

folder <- "D:/ADataBase/flights_data_csv/2024-11-10"
files <- list.files(folder, pattern = "\\.csv$", full.names = TRUE)

for (file in files) {
  
  message("Processing: ", file)
  
  tryCatch({
    
    df <- read_csv(
      file, show_col_types = FALSE
    ) %>%
      filter(
        !is.na(track_longitude),
        !is.na(track_latitude),
        !is.na(track_altitude),
        !is.na(track_timestamp)
      ) %>%
      mutate(
        track_timestamp = as.POSIXct(track_timestamp, format = "%Y.%m.%d, %H:%M:%S"),
        track_longitude = as.numeric(track_longitude),
        track_latitude  = as.numeric(track_latitude),
        track_altitude  = as.numeric(track_altitude),
        track_speed     = as.numeric(track_speed),
        track_heading   = as.numeric(track_heading),
        track_vertical_speed = as.numeric(track_vertical_speed)
      ) %>%
      arrange(identification_number, track_timestamp) %>%
      group_by(identification_number) %>%
      mutate(
        dt = as.numeric(difftime(
          lead(track_timestamp),
          track_timestamp,
          units = "secs"
        )),
        vx = track_speed * sin(track_heading * pi / 180),
        vy = track_speed * cos(track_heading * pi / 180),
        vz = track_vertical_speed
      ) %>%
      ungroup() %>%
      mutate(
        velocity = paste(vx, vy, vz, sep = " "),
        timestamp = as.character(track_timestamp),
        flight_id = identification_number
      )
    
    if (nrow(df) == 0) {
      message("Skipping ", file, ": no valid rows after filtering")
      next
    }
    
    
    geojson_sf <- st_as_sf(
      df,
      coords = c("track_longitude", "track_latitude", "track_altitude"),
      crs = 4326,
      remove = TRUE
    ) %>%
      select(
        geometry,
        velocity,
        timestamp,
        dt,
        flight_id,
        Airline,
        airport_origin_icao
      )
    
    out_file <- file.path(
      "D:/ADataBase/flights_data_geojson/2024-11-10",
      paste0(file_path_sans_ext(basename(file)), "_processed.geojson")
    )
    
    st_write(geojson_sf, out_file, driver = "GeoJSON", append = FALSE)
    message("Written: ", out_file)
    
  }, error = function(e) {
    message("Error in ", file, ": ", e$message)
  })
}

