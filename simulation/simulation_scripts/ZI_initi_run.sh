#!/bin/bash

# different output colors
COLOR='\033[1;36m'          # bold;cyan
COLOR_WARNING='\033[1;33m'  # bold;yellow
COLOR_PROMPT='\033[1;32m'   # bold;green
COLOR_ERROR='\033[1;31m'    # bold;red
NO_COLOR='\033[0m'

# directory from which the script is being run from
CURRENT_DIR="$( pwd )"

function usage
{
    echo -e ${COLOR}
    echo "OVERVIEW: ZI Agents Portfolio Initialization Script"
    echo
    echo "USAGE: ./startup.sh [options] <args>"
    echo
    echo "OPTIONS:";
    echo "  -h [ --help ]                      Display available options"
    echo "  -n [ --seed ] arg                  Random seed path (to store state)"
    echo "  -y [ --repeat ] arg                Repeat the given seed"
    echo "  -k [ --kill ]                      Kill the simulation"
    echo -e ${NO_COLOR}
}

function reset_server
{
    echo -e ${COLOR_WARNING}
    echo "Resetting portfolio and trading records..."
    echo -e ${NO_COLOR}

    cd /home/shiftpub/shift-rl-sim/agents/ZITrader
    # [ -d output ] && rm -r output
    cd ~/SHIFT
    ./startup.sh -m DE ME BC -r </dev/null >/dev/null 2>&1
    ./startup.sh -k </dev/null >/dev/null 2>&1
}

function random_setup {
    #set up the 200 agents' portfolio randomly
    echo -e ${COLOR}
    echo "Creating random initial portfolio values..."
    echo -e ${NO_COLOR}

    # 2-stock scenario
    InitializationProgram -b 1 -e 100 -t CS1 -p 100 -s 10000 </dev/null >/dev/null 2>&1
    InitializationProgram -b 101 -e 200 -t CS2 -p 100 -s 10000 </dev/null >/dev/null 2>&1

    #initialize rl agent:
    cd /home/shiftpub/shift-rl-sim/simulation
    python sim_utils.py

    # go back to directory from which the script is being run from
    cd ${CURRENT_DIR}

    echo -e ${COLOR}
    echo "Done."
    echo -e ${NO_COLOR}
}

function start_sim 
{
    echo -e ${COLOR_WARNING}
    echo "Starting simulation..."
    echo -e ${NO_COLOR}

    # 2-stock scenario
    cd ~/SHIFT
    ./startup.sh
    cd /home/shiftpub/shift-rl-sim/agents/ZITrader
    ./batch2.sh

    # go back to directory from which the script is being run from
    cd ${CURRENT_DIR}

    echo -e ${COLOR}
    echo "Done."
    echo -e ${NO_COLOR}
}

function new_seed_setup
{ 
    #if ( pgrep -x ${1} > /dev/null )
    #then
    echo -e ${COLOR}
    echo "Creating new initial portfolio values with seed ${1}"
    echo -e ${NO_COLOR}

    # 2-stock scenario
    InitializationProgram -n ${1} -b 1 -e 100 -t CS1 -p 100 -s 10000 </dev/null >/dev/null 2>&1
    InitializationProgram -n ${1} -b 101 -e 200 -t CS2 -p 100 -s 10000 </dev/null >/dev/null 2>&1

    #initialize rl agent:
    cd /home/shiftpub/shift-rl-sim/simulation
    python sim_utils.py

    # go back to directory from which the script is being run from
    cd ${CURRENT_DIR}

    echo -e ${COLOR}
    echo "Done."
    echo -e ${NO_COLOR}
    #fi
}

function repeat_seed_setup
{ 
    #if ( pgrep -x ${1} > /dev/null )
    #then
    echo -e ${COLOR}
    echo "Repeat old initial portfolio values with seed ${1}"
    echo -e ${NO_COLOR}

    # 2-stock scenario
    InitializationProgram -y ${1} -b 1 -e 100 -t CS1 -p 100 -s 10000 </dev/null >/dev/null 2>&1
    InitializationProgram -y ${1} -b 101 -e 200 -t CS2 -p 100 -s 10000 </dev/null >/dev/null 2>&1

    #initialize rl agent:
    cd /home/shiftpub/shift-rl-sim/simulation
    python sim_utils.py

    # go back to directory from which the script is being run from
    cd ${CURRENT_DIR}

    echo -e ${COLOR}
    echo "Done."
    echo -e ${NO_COLOR}


    #fi
}

function kill_sim
{
    #cd /home/shiftpub/shift-rl-sim/agents/ZITrader
    # [ -d output ] && rm -r output
    cd ~/SHIFT
    ./startup.sh -k #</dev/null >/dev/null 2>&1
    cd ${CURRENT_DIR}
    echo -e ${COLOR}
    echo "Terminated the server"
    echo -e ${NO_COLOR}

}


while [ "${1}" != "" ]; do
    case ${1} in
        -h | --help )
            usage
            exit 0
            ;;
        -n | --seed )
            reset_server
            new_seed_setup ${2}
            start_sim
            exit 0
            ;;
        -y | --repeat )
            reset_server
            repeat_seed_setup ${2}
            start_sim
            exit 0
            ;;
        -k | --kill )
            kill_sim
            exit 0
            ;;
        *)
            echo
            echo -e "shift: ${COLOR_ERROR}error:${NO_COLOR} ${1} option is not available (please see usage with -h or --help)"
            echo
            exit 1
            ;;
    esac
done

if [ "${1}" == "" ]
then
    reset_server
    random_setup
    start_sim
fi