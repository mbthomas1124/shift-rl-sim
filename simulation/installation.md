Create a new virtual env

```bash
conda create -n rlsim python=3.7
conda activate rlsim
```

Install SHIFT Python Client
```bash
mkdir temp_download
cd temp_download/
wget https://github.com/hanlonlab/shift-python/releases/download/v2.0.1/shift_python-2.0.1-conda_linux.zip
unzip shift_python-2.0.1-conda_linux.zip
cd shift_python-2.0.1-conda_linux/
conda install *
cd ..
rm -rf shift_python-2.0.1-conda_linux/
```

Install Tianshou (with Torch)
```bash
git clone https://github.com/thu-ml/tianshou
cd tianshou
pip install -e .
pip install pandas
```

