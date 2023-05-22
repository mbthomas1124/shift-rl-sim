#!/bin/bash
#
# Simulate Market Maker :  
# Tool to start & stop the 3 components of the Market making Logic
# ZITraders, MarketMaker, ObserverClient
# Usage : smm.sh -reset -time 10min -kill  
#

stopAllZITraders() {
  local f=$1  
  local m
  m=`wc -l $f | sed 's/^\([0-9]*\).*$/\1/'`
  return $m
}

stopMarketMaker() {
  local f=$1
  local m
  m=`wc -l $f | sed 's/^\([0-9]*\).*$/\1/'`
  return $m
}

stopObserverClient() {
  local f=$1
  local m
  m=`wc -l $f | sed 's/^\([0-9]*\).*$/\1/'`
  return $m
}

startAllAgents(){
  cd ~/SHIFT/
  ./startup.sh -o & 
  echo "SHIFT STARTED"

  cd ~/shift-research/agents/ZITrader
  ./batch.sh &
  echo "ZITrader started"	

  cd ~/shift-research/agents/MarketMaker
  ./batch.sh &
  echo "market maker started"

  return 0
  
}



if [ $# -lt 1 ]
then
  echo "Usage: $0 file ..."
  exit 1
fi

echo "Simulate for  $0 minutes" 
l=0
n=0
s=0

startAllAgents

echo " completed "
