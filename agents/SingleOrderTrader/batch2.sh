#!/bin/bash

# delete log & store folders
[ -d log ] && rm -r log
[ -d store ] && rm -r store

# run SingleOrderTraders
for i in {201..210}
do
    nohup ./build/Release/SingleOrderTrader $i CS1 $((1800 + ($i - 200) * 3)) -1 0.9 0 </dev/null >/dev/null 2>&1 &
    # nohup ./build/Release/SingleOrderTrader $i CS1 1800 -1 0.9 0 </dev/null >/dev/null 2>&1 &
done

# run SingleOrderTraders
for i in {211..220}
do
    nohup ./build/Release/SingleOrderTrader $i CS2 $((1800 + ($i - 210) * 3)) -1 0.9 0 </dev/null >/dev/null 2>&1 &
    # nohup ./build/Release/SingleOrderTrader $i CS2 1800 -1 0.9 0 </dev/null >/dev/null 2>&1 &
done
