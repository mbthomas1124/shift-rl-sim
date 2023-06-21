#!/bin/bash

# different output colors
COLOR='\033[1;36m'          # bold;cyan
COLOR_WARNING='\033[1;33m'  # bold;yellow
COLOR_PROMPT='\033[1;32m'   # bold;green
COLOR_ERROR='\033[1;31m'    # bold;red
NO_COLOR='\033[0m'

function setup_robots
{
    # directory from which the script is being run from
    CURRENT_DIR="$( pwd )"

    echo -e ${COLOR_WARNING}
    echo "Resetting portfolio and trading records..."
    echo -e ${NO_COLOR}

    cd /home/shiftpub/shift-research/agents/ZITrader
    # [ -d output ] && rm -r output
    cd ~/SHIFT
    ./startup.sh -m DE ME BC -r </dev/null >/dev/null 2>&1
    ./startup.sh -k </dev/null >/dev/null 2>&1

    echo -e ${COLOR}
    echo "Creating new initial portfolio values..."
    echo -e ${NO_COLOR}

    # # 1-stock scenario
    # InitializationProgram -b 1 -e 200 -t CS1 -p 100 -s 20000 </dev/null >/dev/null 2>&1

    # 2-stock scenario
    InitializationProgram -b 1 -e 100 -t CS1 -p 100 -s 10000 </dev/null >/dev/null 2>&1
    InitializationProgram -b 101 -e 200 -t CS2 -p 100 -s 10000 </dev/null >/dev/null 2>&1

    # # 1-stock market stress agents
    # InitializationProgram -b 201 -e 220 -t CS1 -p 100 -s 2000 </dev/null >/dev/null 2>&1

    # # 2-stock market stress agents
    # InitializationProgram -b 201 -e 210 -t CS1 -p 100 -s 1000 </dev/null >/dev/null 2>&1
    # InitializationProgram -b 211 -e 220 -t CS2 -p 100 -s 1000 </dev/null >/dev/null 2>&1

    # go back to directory from which the script is being run from
    cd ${CURRENT_DIR}

    echo -e ${COLOR}
    echo "Done."
    echo -e ${NO_COLOR}
}

function b_sim_robots
{
    # directory from which the script is being run from
    CURRENT_DIR="$( pwd )"

    echo -e ${COLOR_WARNING}
    echo "Starting simulation..."
    echo -e ${NO_COLOR}

    # # 1-stock scenario
    # cd ~/SHIFT
    # ./startup.sh
    # cd /home/shiftpub/shift-research/agents/ZITrader
    # ./batch.sh

    # 2-stock scenario
    cd ~/SHIFT
    ./startup.sh
    cd /home/shiftpub/shift-research/agents/ZITrader
    ./batch2.sh

    # # 1-stock market stress agents
    # cd /home/shiftpub/shift-research/agents/SingleOrderTrader
    # ./batch.sh

    # # 2-stock market stress agents
    # cd /home/shiftpub/shift-research/agents/SingleOrderTrader
    # ./batch2.sh

    # go back to directory from which the script is being run from
    cd ${CURRENT_DIR}

    echo -e ${COLOR}
    echo "Done."
    echo -e ${NO_COLOR}
}

function e_sim_robots
{
    # directory from which the script is being run from
    CURRENT_DIR="$( pwd )"

    echo -e ${COLOR_WARNING}
    echo "Stoping simulation..."
    echo -e ${NO_COLOR}

    # # market stress agents
    # killall -9 SingleOrderTrader

    killall -9 ZITrader
    cd ~/SHIFT
    ./startup.sh -k

    # go back to directory from which the script is being run from
    cd ${CURRENT_DIR}

    echo -e ${COLOR}
    echo "Done."
    echo -e ${NO_COLOR}
}
