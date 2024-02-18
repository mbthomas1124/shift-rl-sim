#!/bin/bash

total_runs=$1

for (( run_num=1; run_num<=$total_runs; run_num++ ))
do
    cd ~/shift-rl-sim/simulation/simulation_scripts
    ./initial_sim.sh
    wait

    cd ~/shift-rl-sim/simulation
    python ppo_parallel.py $run_num
    wait

    echo "Run $run_num completed."
done