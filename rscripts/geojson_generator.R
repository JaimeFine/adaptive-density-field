library(dplyr)
library(jsonlite)
library(purrr)
library(readr)
library(tools)

TIMESTAMP_FORMAT <- "%Y-%m-%d %H:%M:%S"

folder <- "C:/Users/13647/OneDrive/Desktop"
files <- list.files(folder, pattern = "\\.csv$", full.names=TRUE)

for (file in files) {
  
  message("Processing: ", file)

  tryCatch({
    # Basic processing:
    geojson <- read_csv(
      file,
      col_types = cols(
        track_timestamp = col_datetime(format = "%Y.%m.%d, %H:%M:%S"),
        track_longitude = col_double(),
        track_latitude = col_double(),
        track_altitude = col_double(),
        track_speed = col_double(),
        track_heading = col_double(),
        track_vertical_speed = col_double(),
        identification_number = col_character(),
        Airline = col_character(),
        airport_origin_icao = col_character()
      )
    ) %>%
      filter(
        !is.na(track_longitude),
        !is.na(track_latitude),
        !is.na(track_altitude),
        !is.na(track_timestamp)
      )
    
    # Adding new elements:
    geojson <- geojson %>%
      arrange(identification_number, track_timestamp) %>%
      group_by(identification_number) %>%
      mutate(
        dt = as.numeric(
          difftime(
            lead(track_timestamp),
            track_timestamp,
            units = "secs"
          )
        )
      ) %>%
      mutate(
        vx = track_speed * sin(track_heading * pi / 180),
        vy = track_speed * cos(track_heading * pi / 180),
        vz = track_vertical_speed
      ) %>% ungroup()
    
    state_list <- lapply(1:nrow(geojson), function(i) {
      list(
        pos = c(geojson$track_longitude[i], geojson$track_latitude[i], geojson$track_altitude[i]),
        vel = c(geojson$vx[i], geojson$vy[i], geojson$vz[i])
      )
    })
    
    geojson <- geojson %>%
      mutate(
        state = purrr::pmap(
          list(
            track_longitude,
            track_latitude,
            track_altitude,
            vx, vy, vz
          ),
          function(lon, lat, alt, vx, vy, vz) {
            list(
              pos = c(lon, lat, alt),
              vel = c(vx, vy, vz)
            )
          }
        )
      )
    
    # Creating .geojson file:
    features <- vector("list", nrow(geojson))
    for (i in seq_len(nrow(geojson))) {
      features[[i]] <- list(
        type = "Feature",
        geometry = list(
          type = "Point",
          coordinates = c(geojson$track_longitude[i], geojson$track_latitude[i], geojson$track_altitude[i])
        ),
        properties = list(
          state = state_list[[i]],
          timestamp = as.character(geojson$track_timestamp[i]),
          dt = geojson$dt[i],
          flight_id = geojson$identification_number[i],
          Airline = geojson$Airline[i],
          airport_origin_icao = geojson$airport_origin_icao[i]
        )
      )
    }
    
    geojson_out <- list(type = "FeatureCollection", features = features)
    
    out_file <- file.path(folder, paste0(file_path_sans_ext(basename(file)), "_processed.geojson"))
    
    jsonlite::write_json(geojson_out, out_file, auto_unbox = TRUE)
    
    print("Written")
    
  }, error = function(e) {
    message("Error in ", file, ": ", e$message)
  })
}

