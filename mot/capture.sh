#!/bin/bash

# Get the current date and time in the format YYYYMMDD-HHMMSS
DATE=$(date +"%Y%m%d-%H%M%S")

# Use the date and time to create a filename
FILENAME="video-${DATE}.mkv"

# https://gstreamer.freedesktop.org/documentation/applemedia/vtenc_h264.html
# gst-device-monitor-1.0
gst-launch-1.0 -e \
  avfvideosrc device-index=0 ! \
  video/x-raw,format=UYVY,width=640,height=480,framerate=3000003/100000 ! \
  videoconvert ! \
  x264enc ! \
  matroskamux ! \
  filesink location=$FILENAME


# FILENAME="video-${DATE}.mov"
# gst-launch-1.0 -e \
#   avfvideosrc device-index=0 ! \
#   video/x-raw,format=UYVY,width=640,height=480,framerate=3000003/100000 ! \
#   videoconvert ! \
#   vtenc_h264 ! \
#   qtmux ! \
#   filesink location=$FILENAME
