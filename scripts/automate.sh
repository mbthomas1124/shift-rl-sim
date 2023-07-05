#!/bin/bash

# different output colors
COLOR='\033[1;36m'          # bold;cyan
COLOR_WARNING='\033[1;33m'  # bold;yellow
COLOR_PROMPT='\033[1;32m'   # bold;green
COLOR_ERROR='\033[1;31m'    # bold;red
NO_COLOR='\033[0m'



function run_ppo
{
    if [ $# -ne 2 ]
    then
        echo "[${1}] Incorrect number of arguments supplied"
        return 1
    fi
    
    echo -e ${COLOR_WARNING}
    echo "[${1}] Start Simulation..."
    echo -e ${NO_COLOR}

    ssh shiftpub@${1} "
        . /home/shiftpub/miniconda/etc/profile.d/conda.sh;
        conda activate rlsim;
        cd ~/shift-rl-sim/simulation/simulation_scripts;
        ./initial_sim.sh
    "

    echo -e ${COLOR}
    echo "[${1}] Start Training..."
    echo -e ${NO_COLOR}


    ssh shiftpub@${1} "
        . /home/shiftpub/miniconda/etc/profile.d/conda.sh;
        conda activate rlsim;
        cd ~/shift-rl-sim/simulation;
        nohup python -u ppo_parallel.py > ${2} &
    "

    echo -e ${COLOR}
    echo "[${1}] Done."
    echo -e ${NO_COLOR}
}

function kill_all
{
    if [ $# -ne 1 ]
    then
        echo "Incorrect number of arguments supplied"
        return 1
    fi
    
    echo -e ${COLOR_WARNING}
    echo "[${1}] Kill all Python..."
    echo -e ${NO_COLOR}

    ssh shiftpub@${1} "
        killall python
    "

    echo -e ${COLOR}
    echo "[${1}] Kill SHIFT on ${1}"
    echo -e ${NO_COLOR}


    ssh shiftpub@${1} "
        cd ~/SHIFT
        ./startup.sh -k
    "

    echo -e ${COLOR}
    echo "[${1}] Done."
    echo -e ${NO_COLOR}
}

function send_dir
{
    # arg 1: remote IP; arg 2: path to local dir
    # send the directory to the same location on remote machine. 
    if [ $# -ne 2 ]
    then
        echo "[${1}] Incorrect number of arguments supplied"
        return 1
    fi
    
    echo -e ${COLOR_WARNING}
    echo "[${1}] Send file from ${2} to shiftpub@${1}:${2} ..."
    echo -e ${NO_COLOR}

    ssh shiftpub@${1} "
    if [ ! -d ${2} ]
    then
        echo "[${1}] Creating dir on remote shiftpub@${1}:${2}"
        mkdir -p ${2}
    fi
    "

    rsync -r ${2}/* shiftpub@${1}:${2}

    echo -e ${COLOR}
    echo "[${1}] Done."
    echo -e ${NO_COLOR}
}


function fetch_dir
{
    # fetch the directory to the current machine at ~/receive/{ip}/
    if [ $# -ne 2 ]
    then
        echo "[${1}] Incorrect number of arguments supplied"
        return 1
    fi
    
    echo -e ${COLOR}
    echo "[${1}] Fetch file from shiftpub@${1}:${2} to ~/receive/${1}/..."
    echo -e ${NO_COLOR}

    [ ! -d /home/shiftpub/receive/${1}/ ] && (echo -e ${COLOR_WARNING}; echo "Directory "~/receive/${1}/" DOES NOT exists. Making the directory"; mkdir -p ~/receive/${1}/;echo -e ${NO_COLOR};)
    
    [ ! -d /home/shiftpub/receive/155.246.104.85/ ]
    rsync -r shiftpub@${1}:${2} ~/receive/${1}/

    echo -e ${COLOR}
    echo "[${1}] Done."
    echo -e ${NO_COLOR}
}

function check_training
{
    if [ $# -ne 2 ]
    then
        echo "[${1}] Incorrect number of arguments supplied"
        return 1
    fi
    echo -e ${COLOR}
    echo "[${1}] Check SHIFT status"
    echo -e ${NO_COLOR}

    ssh shiftpub@${1} "
        cd ~/SHIFT;
        ./startup.sh -s;
    "

    echo -e ${COLOR}
    echo "[${1}] Check Python outputs"
    echo -e ${NO_COLOR}


    ssh shiftpub@${1} "
        tail -n 100 ${2}
    "
    # print SHIFT status
    # print tail logs python
    # print python process
}

