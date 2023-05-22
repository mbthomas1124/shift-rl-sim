#!/bin/bash

# delete log & store folders
[ -d log ] && rm -r log
[ -d store ] && rm -r store

# create output directory if it does not exist yet
[ -d output ] || mkdir output

# run ObserverClient writing in output (23400s == 390min)
nohup ./build/Release/ObserverClient 0 ? 0 5 23400.0 0.5 output 0 </dev/null >/dev/null 2>&1 &
