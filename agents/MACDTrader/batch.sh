#!/bin/bash

# delete log & store folders
[ -d log ] && rm -r log
[ -d store ] && rm -r store

for i in {201..210}
do
    # nohup ./build/Release/MACDTrader $i CS1 22500 10 34 55 12 1 0.02 0.06 0 0 0 0 0 </dev/null >/dev/null 2>&1 &
    nohup ./build/Release/MACDTrader $i CS1 3900 5 34 55 12 1 0.02 0.06 0 0 0 0 0 </dev/null >/dev/null 2>&1 &
done
