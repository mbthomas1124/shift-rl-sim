import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from numpy import round

def episodeReward(df):
    '''
    df: pandas dataframe
    :return: list of reward
    '''
    agg_cost = []
    obj_shares = []
    tot_rwd = 0
    obj_share = df.loc[0, 'remained_shares']
    for i in df.index:
        tot_rwd += df.loc[i, "reward"]
        if df.loc[i, "done"]:
            agg_cost.append(tot_rwd)
            obj_shares.append(obj_share)
            try:
                obj_share = df.loc[i + 1, 'remained_shares']
            except KeyError:
                break
            tot_rwd = 0
    return np.array(agg_cost), np.array(obj_shares)

def aggData(path):
    realPath = os.path.join(path, 'iteration_info')
    files = os.listdir(realPath)
    ret = pd.read_csv(os.path.join(realPath, files[0]), index_col = 0)
    for oneTab in files[1:]:
        tmp = pd.read_csv(os.path.join(realPath, oneTab), index_col = 0 )
        ret = pd.concat([ret, tmp], ignore_index = True)
    return ret



if __name__ == '__main__':
    chkpt_path = '2019-07-30_15-46-56'
    full_tab = aggData(chkpt_path)
    agg_reward, shares = episodeReward(full_tab)
