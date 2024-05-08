#!/bin/bash

# Get the current date and time in the format YYYYMMDD-HHMMSS
DATE=$(date +"%Y%m%d-%H%M%S")

# Use the date and time to create a filename
FILENAME="video-${DATE}.mkv"

gst-launch-1.0 -e \
  avfvideosrc device-index=1 ! \
  videoconvert ! \
  x264enc ! \
  matroskamux ! \
  filesink location=$FILENAME
