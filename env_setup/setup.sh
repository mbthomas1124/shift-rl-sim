# directory from which the script is being run from
CURRENT_DIR="$( pwd )"

cd ~/shift-rl-sim

# create conda environment
conda create -n rlsim python=3.7
conda activate rlsim
mkdir temp_download
cd temp_download/
wget https://github.com/hanlonlab/shift-python/releases/download/v2.0.1/shift_python-2.0.1-conda_linux.zip
unzip shift_python-2.0.1-conda_linux.zip
cd shift_python-2.0.1-conda_linux/
conda install *
cd ../..
rm -rf temp_download

# install pip packages
pip install -r requirements.txt

# create agents for simulations
env_setup/setup.sh

