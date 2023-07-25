#!/bin/bash
eval "$(conda shell.bash hook)"

# directory from which the script is being run from
CURRENT_DIR="$( pwd )"

cd ~/shift-rl-sim

# create conda environment
mkdir temp_download
cd temp_download/
conda create -n sep_rlsim python=3.7 -y
conda clean --all -y
conda activate sep_rlsim

# install shift
wget https://github.com/hanlonlab/shift-python/releases/download/v2.0.1/shift_python-2.0.1-conda_linux.zip
unzip -j shift_python-2.0.1-conda_linux.zip
conda install *.bz2 -y
conda update --all -y
conda clean --all -y
cd ~/shift-rl-sim
rm -rf temp_download

# install pip packages
pip install -r requirements.txt

# create agents for simulations
cd ~/shift-rl-sim/env_setup
./new_agents.sh

# verify that python env was setup properly
python test_env.py

cd ${CURRENT_DIR}