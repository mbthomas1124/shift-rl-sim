#!/bin/bash

# delete log & store folders
[ -d log ] && rm -r log
[ -d store ] && rm -r store

# create output directories if they do not exist yet
[ -d output ] || mkdir output
[ -d rng ] || mkdir rng

#run ZITraders (23100s == 385min)
for i in {1..100}
do
    nohup ./build/Release/ZITrader $i CS1 23100.0 385.0 2 2 100.00 0.10 0.01 0 0 </dev/null >/dev/null 2>&1 &
done

# for i in {101..200}
# do
#     nohup ./build/Release/ZITrader $i CS2 23100.0 385.0 2 2 100.00 0.10 0.01 0 0 </dev/null >/dev/null 2>&1 &
# done
