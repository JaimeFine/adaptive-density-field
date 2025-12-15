library(dplyr)
library(readr)

# All airport_destination_icao = ZUUU
# All time_estimated_arrival = NA
# All airport_destination_longitude = 103.947
# All airport_destination_latitude = 30.57852
# All airport_destination_offset = 28800
# All time_estimated_departure = ""
raw <- read.csv("shuangliu.csv", header=TRUE)
flights <- raw %>%
  filter(!is.na(time_other_duration)) %>%
  select(
    -time_estimated_arrival,
    -time_estimated_departure,
    -airport_destination_icao,
    -airport_destination_longitude,
    -airport_destination_latitude,
    -airport_destination_offset
)

write_csv("cleaned.csv")