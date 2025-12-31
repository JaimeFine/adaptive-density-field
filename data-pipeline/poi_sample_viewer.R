library(ggplot2)
library(dplyr)
library(leaflet)
library(leaflet.extras)
library(MASS)

pois <- read.csv("poi_data.csv")

set.seed(42)
pois_vis <- pois %>% sample_n(min(100000, n()))

# Sample view with Heat map
leaflet(pois_vis) %>%
  addProviderTiles("CartoDB.Positron") %>%
  setView(lng = 104.0667, lat = 30.6667, zoom = 3) %>%
  addHeatmap(
    lng = ~lon,
    lat = ~lat,
    intensity = ~poi_score,
    blur = 10,
    max = max(pois_vis$poi_score, na.rm = TRUE)
  )

leaflet(pois_vis) %>%
  addTiles() %>%
  setView(lng = 104.0667, lat = 30.6667, zoom = 3) %>%
  addCircleMarkers(
    color = "red", fillColor = "orange",
    fillOpacity = 0.4, radius = 1,
    popup = "POI"
  )

# Sample view with KDE algorithm
x <- pois$lon
y <- pois$lat
w <- pois$poi_score

h <- c(0.02, 0.02) 

kde <- kde2d(
  x, y,
  n = 200,
  lims = c(range(x), range(y)),
)

z_weighted <- matrix(0, nrow = nrow(kde$z), ncol = ncol(kde$z))
for (i in seq_along(x)) {
  z_weighted <- z_weighted +
    w[i] * outer(
      dnorm(kde$x, x[i], h[1]),
      dnorm(kde$y, y[i], h[2])
    )
}

persp(
  kde$x, kde$y, z_weighted,
  theta = 30, phi = 30,
  col = "lightgreen",
  shade = 0.01,
  xlab = "Longitude",
  ylab = "Latitude",
  zlab = "Density"
)

