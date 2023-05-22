#!/bin/bash

# delete log & store folders
[ -d log ] && rm -r log
[ -d store ] && rm -r store

# create output directory if it does not exist yet
[ -d output ] || mkdir output

# run MarketMaker writing in output
nohup ./build/Release/MarketMaker CS1 3600 2 0.05 output 0 </dev/null >/dev/null 2>&1 &
