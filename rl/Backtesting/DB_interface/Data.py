import pandas as pd
import os
# pd.read_csv('data/tmp.csv', index_col=0)
cwd = os.getcwd()

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
Open = pd.read_csv('data/Open.csv', index_col=0)
Close = pd.read_csv('data/Close.csv', index_col=0)
High = pd.read_csv('data/High.csv', index_col=0)
Low = pd.read_csv('data/Low.csv', index_col=0)
Volume = pd.read_csv('data/Volume.csv', index_col=0)
# SPY = pd.read_csv('data/SP.csv', index_col = 0)
Adj_Close = pd.read_csv('data/Adj Close.csv', index_col = 0 )

# Calendar = pd.read_csv('data/tmp.csv', index_col=0)
os.chdir(cwd)

