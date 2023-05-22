#!/bin/bash
set -m

# directory from which the script is being run from
CURRENT_DIR="$( pwd )"

# output colors
COLOR_PROMPT='\033[1;32m'   # bold;green
COLOR_ERROR='\033[1;31m'    # bold;red
NO_COLOR='\033[0m'



# simulation parameters
NUMTRIALS=1
TRIALTIME=1
ACTION=0
LAMBDA=1
ORDERSIZE=10
ORDERLISTLEN=20

# start simulation
cd ~/shift-research/rl/RL_with_ZI/simulation_scripts
./ZI_initi_run.sh -n 1
sleep 15

for (( TRIAL=0; TRIAL<$NUMTRIALS; TRIAL++ ))
do

# make sure that Brokerage Center is running
SERVICE="BrokerageCenter"
if ( pgrep -x $SERVICE > /dev/null )
    then
        echo
        echo -ne "shift: $SERVICE status: ${COLOR_PROMPT}running${NO_COLOR}"
        echo
    else
        echo
        echo -ne "shift: $SERVICE status: ${COLOR_ERROR}not running${NO_COLOR}"
        echo
        # kill simulation and return to directory from which script was ran
        cd ~/shift-research/rl/RL_with_ZI/simulation_scripts
        ./ZI_initi_run.sh -k
        cd ${CURRENT_DIR}
        exit 1
    fi

# collect data with only ZI agents and no RL
cd ~/shift-research/rl/RL_with_ZI
python test_LOB.py $ACTION $LAMBDA $ORDERSIZE $ORDERLISTLEN $TRIAL $TRIALTIME 0

# set up RL environment
cd ~/shift-research/rl/RL_with_ZI
python mult_env_test.py $ACTION $LAMBDA $ORDERSIZE $ORDERLISTLEN $TRIALTIME >/dev/null &
sleep 15

# collect data with ZI agents and RL
cd ~/shift-research/rl/RL_with_ZI
python test_LOB.py $ACTION $LAMBDA $ORDERSIZE $ORDERLISTLEN $TRIAL $TRIALTIME 1
fg

if [ $TRIAL -lt $(( NUMTRIALS - 1 )) ]
then
# restart simulation
cd ~/shift-research/rl/RL_with_ZI/simulation_scripts
./ZI_initi_run.sh -y 1
sleep 15
fi

done

# kill simulation and return to directory from which script was ran
cd ~/shift-research/rl/RL_with_ZI/simulation_scripts
./ZI_initi_run.sh -k
cd ${CURRENT_DIR}