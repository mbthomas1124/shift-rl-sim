#!/bin/bash

# delete log & store folders
[ -d log ] && rm -r log
[ -d store ] && rm -r store

# create output directory if it does not exist yet
[ -d output ] || mkdir output

# run RhoTraders (23400s == 390min)
for i in {201..201}
do
    nohup ./build/Release/RhoTrader $i CS1 CS2 23400.0 1.0 300 1 1 0.5 1.0 2.0 0 </dev/null >/dev/null 2>&1 &
done
