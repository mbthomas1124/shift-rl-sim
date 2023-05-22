#!/bin/bash

# delete log & store folders
[ -d log ] && rm -r log
[ -d store ] && rm -r store

for i in {201..201} # Last - Signal - Limit
do
    # nohup ./build/Release/MACDTrader $i CS1 22500 10 34 55 12 1 0.02 0.06 0 0 1 0 1 </dev/null >/dev/null 2>&1 &
    nohup ./build/Release/MACDTrader $i CS1 3900 5 34 55 12 1 0.02 0.06 0 0 1 0 1 </dev/null >/dev/null 2>&1 &
done

for i in {202..202} # Last - Hist. - Market
do
    # nohup ./build/Release/MACDTrader $i CS1 22500 10 34 55 12 1 0.02 0.06 0 1 0 0 1 </dev/null >/dev/null 2>&1 &
    nohup ./build/Release/MACDTrader $i CS1 3900 5 34 55 12 1 0.02 0.06 0 1 0 0 1 </dev/null >/dev/null 2>&1 &
done

for i in {203..203} # Last - Hist. (x0.5) - Limit
do
    # nohup ./build/Release/MACDTrader $i CS1 22500 10 34 55 12 1 0.02 0.06 0 1 1 0 1 0.5 </dev/null >/dev/null 2>&1 &
    nohup ./build/Release/MACDTrader $i CS1 3900 5 34 55 12 1 0.02 0.06 0 1 1 0 1 0.5 </dev/null >/dev/null 2>&1 &
done

for i in {204..204} # Last - Hist. (x1) - Limit
do
    # nohup ./build/Release/MACDTrader $i CS1 22500 10 34 55 12 1 0.02 0.06 0 1 1 0 1 1.0 </dev/null >/dev/null 2>&1 &
    nohup ./build/Release/MACDTrader $i CS1 3900 5 34 55 12 1 0.02 0.06 0 1 1 0 1 1.0 </dev/null >/dev/null 2>&1 &
done

for i in {205..205} # Last - Hist. (x2) - Limit
do
    # nohup ./build/Release/MACDTrader $i CS1 22500 10 34 55 12 1 0.02 0.06 0 1 1 0 1 2.0 </dev/null >/dev/null 2>&1 &
    nohup ./build/Release/MACDTrader $i CS1 3900 5 34 55 12 1 0.02 0.06 0 1 1 0 1 2.0 </dev/null >/dev/null 2>&1 &
done

for i in {206..206} # Mid - Signal - Limit
do
    # nohup ./build/Release/MACDTrader $i CS1 22500 10 34 55 12 1 0.02 0.06 1 0 1 0 1 </dev/null >/dev/null 2>&1 &
    nohup ./build/Release/MACDTrader $i CS1 3900 5 34 55 12 1 0.02 0.06 1 0 1 0 1 </dev/null >/dev/null 2>&1 &
done

for i in {207..207} # Mid - Hist. - Market
do
    # nohup ./build/Release/MACDTrader $i CS1 22500 10 34 55 12 1 0.02 0.06 1 1 0 0 1 </dev/null >/dev/null 2>&1 &
    nohup ./build/Release/MACDTrader $i CS1 3900 5 34 55 12 1 0.02 0.06 1 1 0 0 1 </dev/null >/dev/null 2>&1 &
done

for i in {208..208} # Mid - Hist. (x0.5) - Limit
do
    # nohup ./build/Release/MACDTrader $i CS1 22500 10 34 55 12 1 0.02 0.06 1 1 1 0 1 0.5 </dev/null >/dev/null 2>&1 &
    nohup ./build/Release/MACDTrader $i CS1 3900 5 34 55 12 1 0.02 0.06 1 1 1 0 1 0.5 </dev/null >/dev/null 2>&1 &
done

for i in {209..209} # Mid - Hist. (x1) - Limit
do
    # nohup ./build/Release/MACDTrader $i CS1 22500 10 34 55 12 1 0.02 0.06 1 1 1 0 1 1.0 </dev/null >/dev/null 2>&1 &
    nohup ./build/Release/MACDTrader $i CS1 3900 5 34 55 12 1 0.02 0.06 1 1 1 0 1 1.0 </dev/null >/dev/null 2>&1 &
done

for i in {210..210} # Mid - Hist. (x2) - Limit
do
    # nohup ./build/Release/MACDTrader $i CS1 22500 10 34 55 12 1 0.02 0.06 1 1 1 0 1 2.0 </dev/null >/dev/null 2>&1 &
    nohup ./build/Release/MACDTrader $i CS1 3900 5 34 55 12 1 0.02 0.06 1 1 1 0 1 2.0 </dev/null >/dev/null 2>&1 &
done
