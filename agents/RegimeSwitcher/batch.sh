#!/bin/bash

# delete log & store folders
[ -d log ] && rm -r log
[ -d store ] && rm -r store

# create output directory if it does not exist yet
[ -d output ] || mkdir output

# run RegimeSwitcher writing in output
nohup ./build/Release/RegimeSwitcher CS1 3600 2 0.05 output 0 </dev/null >regime_log.txt 2>&1 &
