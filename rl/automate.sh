#!/bin/bash

# different output colors
COLOR='\033[1;36m'          # bold;cyan
COLOR_WARNING='\033[1;33m'  # bold;yellow
COLOR_PROMPT='\033[1;32m'   # bold;green
COLOR_ERROR='\033[1;31m'    # bold;red
NO_COLOR='\033[0m'

function setup
{
    if [ $# -ne 2 ]
    then
        echo "Incorrect number of arguments supplied"
        return 1
    fi
    
    echo -e ${COLOR_WARNING}
    echo "Resetting portfolio and trading records..."
    echo -e ${NO_COLOR}

    ssh ${1}@155.246.104.${2} "
        cd ~/shift-research/agents/ObserverClient
        [ -d output ] && rm -r output

        cd ~/shift-research/agents/ZITrader
        [ -d output ] && rm -r output

        cd ~/shift-main
        ./startup.sh -m DE ME BC -r </dev/null >/dev/null 2>&1
        ./startup.sh -k </dev/null >/dev/null 2>&1
    "

    echo -e ${COLOR}
    echo "Creating new initial portfolio values..."
    echo -e ${NO_COLOR}

    ssh ${1}@155.246.104.${2} "
        InitializationProgram -b 1 -e 200 -t CS1 -p 100 -s 20000 </dev/null >/dev/null 2>&1
    "

    echo -e ${COLOR}
    echo "Done."
    echo -e ${NO_COLOR}
}

function b_sim
{
    if [ $# -ne 2 ]
    then
        echo "Incorrect number of arguments supplied"
        return 1
    fi

    # directory from which the script is being run from
    CURRENT_DIR="$( pwd )"

    # create new_experiment folder if it does not exist
    [ -d new_experiment_${2} ] || mkdir ${CURRENT_DIR}/new_experiment_${2}

    echo -e ${COLOR}
    echo "Exporting traders information..."
    echo -e ${NO_COLOR}

    psql -U hanlonpgsql4 -q -h 155.246.104.${2} -d shift_brokeragecenter -c "\COPY traders TO '${CURRENT_DIR}/new_experiment_${2}/traders.csv' CSV HEADER"

    echo -e ${COLOR}
    echo "Exporting initial portfolio values..."
    echo -e ${NO_COLOR}

    psql -U hanlonpgsql4 -q -h 155.246.104.${2} -d shift_brokeragecenter -c "\COPY portfolio_summary TO '${CURRENT_DIR}/new_experiment_${2}/b_portfolio_summary.csv' CSV HEADER"
    psql -U hanlonpgsql4 -q -h 155.246.104.${2} -d shift_brokeragecenter -c "\COPY portfolio_items TO '${CURRENT_DIR}/new_experiment_${2}/b_portfolio_items.csv' CSV HEADER"

    echo -e ${COLOR_WARNING}
    echo "Starting simulation..."
    echo -e ${NO_COLOR}

    ssh ${1}@155.246.104.${2} "
        cd ~/shift-main
        ./startup.sh -m DE ME BC

        cd ~/shift-research/agents/ZITrader
        ./batch.sh

        cd ~/shift-research/agents/ObserverClient
        ./batch.sh
    "

    echo -e ${COLOR}
    echo "Done."
    echo -e ${NO_COLOR}
}

function e_sim
{
    if [ $# -ne 2 ]
    then
        echo "Incorrect number of arguments supplied"
        return 1
    fi

    # directory from which the script is being run from
    CURRENT_DIR="$( pwd )"

    # create new_experiment folder if it does not exist
    [ -d new_experiment_${2} ] || mkdir ${CURRENT_DIR}/new_experiment_${2}

    echo -e ${COLOR_WARNING}
    echo "Stoping simulation..."
    echo -e ${NO_COLOR}

    ssh ${1}@155.246.104.${2} "
        killall -9 ObserverClient

        killall -9 ZITrader

        cd ~/shift-main
        ./startup.sh -k
    "

    echo -e ${COLOR}
    echo "Exporting final portfolio values..."
    echo -e ${NO_COLOR}

    psql -U hanlonpgsql4 -q -h 155.246.104.${2} -d shift_brokeragecenter -c "\COPY portfolio_summary TO '${CURRENT_DIR}/new_experiment_${2}/e_portfolio_summary.csv' CSV HEADER"
    psql -U hanlonpgsql4 -q -h 155.246.104.${2} -d shift_brokeragecenter -c "\COPY portfolio_items TO '${CURRENT_DIR}/new_experiment_${2}/e_portfolio_items.csv' CSV HEADER"

    echo -e ${COLOR}
    echo "Exporting trading records..."
    echo -e ${NO_COLOR}

    psql -U hanlonpgsql4 -q -h 155.246.104.${2} -d shift_brokeragecenter -c "\COPY trading_records TO '${CURRENT_DIR}/new_experiment_${2}/trading_records.csv' CSV HEADER"

    echo -e ${COLOR}
    echo "Retrieving ObserverClient output..."
    echo -e ${NO_COLOR}

    scp -r ${1}@155.246.104.${2}:~/shift-research/agents/ObserverClient/output new_experiment_${2}/.
    for old in new_experiment_${2}/output/CS*.csv; do
        new=$(echo $old | sed -e 's/2019.*/price_records.csv/')
        mv "$old" "$new"
    done
    mv new_experiment_${2}/output/* new_experiment_${2}/

    echo -e ${COLOR}
    echo "Retrieving ZITrader output..."
    echo -e ${NO_COLOR}

    scp -r ${1}@155.246.104.${2}:~/shift-research/agents/ZITrader/output new_experiment_${2}/.
    for old in new_experiment_${2}/output/agent*.csv; do
        new=$(echo $old | sed -e 's/2019.*/wealth_records.csv/')
        mv "$old" "$new"
    done
    [ -d new_experiment_${2}/agents_data ] || mkdir new_experiment_${2}/agents_data
    mv new_experiment_${2}/output/* new_experiment_${2}/agents_data/

    # delete empty output folder
    rm -r new_experiment_${2}/output

    echo -e ${COLOR}
    echo "Done."
    echo -e ${NO_COLOR}
}