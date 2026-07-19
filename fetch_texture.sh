#!/bin/bash
# Fetches the real lunar albedo texture used by libration_field_v4.py.
#
# Source: mrdoob/three.js examples/textures/planets/moon_1024.jpg
# An equirectangular, nearside-centered lunar mosaic derived from the
# Clementine spacecraft dataset (USGS Map-a-Planet / PDS, public domain),
# redistributed in the three.js repo (MIT-licensed project; the imagery
# itself is US-government public domain).
#
# For higher resolution or true LOLA elevation data, replace this with:
#   - USGS Astrogeology Science Center (astrogeology.usgs.gov) LOLA DEM
#   - NASA PDS Geosciences Node lunar products
# and adjust the sampling code in libration_field_v4.py accordingly
# (it already expects a single-band or RGB equirectangular raster with
# column 0..W-1 = longitude -180..+180, row 0..H-1 = latitude +90..-90).

curl -sL -o moon_1024.jpg \
  "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/moon_1024.jpg"
