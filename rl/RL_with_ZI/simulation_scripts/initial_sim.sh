#!/bin/bash

# directory from which the script is being run from
CURRENT_DIR="$( pwd )"

#num_ticker = 6

echo -e ${COLOR_WARNING}
echo "Resetting portfolio and trading records..."
echo -e ${NO_COLOR}

cd ~/SHIFT
./startup.sh -r </dev/null >/dev/null 2>&1 # -m DE ME BC
./startup.sh -k </dev/null >/dev/null 2>&1

#cd /home/shiftpub/shift-research/agents/ZITrader
#./batch_ultimate.sh -t 10 -a 50 -s 5
#timeout -t 10

echo -e ${COLOR}
echo "Repeat old initial portfolio values with seed 1"
echo -e ${NO_COLOR}

#intitializationProgram
# start=1
# end=100
# step=100
# for i in $(seq 1 1 2)
# do
#    InitializationProgram -n 1 -b $start -e $end -t "CS$i" -p 100 -s 10000 </dev/null >/dev/null 2>&1
#    ((start+=$step))
#    ((end+=$step))
# done
InitializationProgram -n 5 -b 1 -e 100 -t CS1 -p 100 -s 10000 </dev/null >/dev/null 2>&1
#InitializationProgram -n 5 -b 101 -e 200 -t CS2 -p 100 -s 10000 </dev/null >/dev/null 2>&1
#InitializationProgram -y 5 -b 201 -e 300 -t CS3 -p 100 -s 10000 </dev/null >/dev/null 2>&1

#initialize rl agent:
cd /home/shiftpub/shift-research/rl/RL_with_ZI
python sim_utils.py

cd ~/SHIFT
./startup.sh

#sleep 5
cd /home/shiftpub/shift-research/agents/ZITrader
#./batch.sh
#./batch_ultimate.sh -t 2 -a 100 -s 5
echo -e ${COLOR}
echo "Done"
echo -e ${NO_COLOR}

cd ${CURRENT_DIR}