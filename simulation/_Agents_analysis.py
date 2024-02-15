import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import statsmodels
from typing import Tuple, List
import scipy
import math
import os



def __save_image(filename):
    # PdfPages is a wrapper around pdf
    # file so there is no clash and create
    # files with no error.
    p = PdfPages(filename)

    # get_fignums Return list of existing
    # figure numbers
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]

    # iterating over the numbers in list
    for fig in figs:
        # and saving the files
        fig.savefig(p, format="pdf")

    # close the object
    p.close()


# function for mapping the pdf
def map_pdf(x, ax, **kwargs):
    mu = x.mean()
    sig = x.std()
    x0, x1 = ax.get_xlim()  # extract the endpoints for the x-axis
    x_pdf = np.linspace(x0, x1, 100)
    plt.plot(x_pdf, scipy.stats.norm.pdf(x_pdf, mu, sig), "r", lw=2, label="pdf")

def find_max_min(data_list, num_agents):
    gap_ratio = 0.1
    maxlist = []
    minlist = []
    for i in range(num_agents):
        maxlist.append(data_list[i].max())
        minlist.append(data_list[i].min())
    Min = min(minlist)
    Max = max(maxlist)
    average = (abs(Min)+abs(Max))/2
    return (Min - (average*gap_ratio), Max + (average*gap_ratio))

def get_MM_stats(num_MM, mov_avg_frame, folder_num):
    # PARAMS:
    # num_MM: number of MM agents
    # filename: the name of the resulting pdf file

    plt.rcParams["figure.figsize"] = [12, 3.50]
    plt.rcParams["figure.autolayout"] = True

    add = 0
    if folder_num == 2:
        add = 4
    data_list = []
    for i in range(num_MM):
        data_list.append(pd.read_csv(f"sep_trader_mm{i+1+add}.csv"))
    
    #Average reward comparison:
    plt.figure()
    temp_list = []
    for i in range(num_MM):
        temp_list.append(data_list[i]["reward"].rolling(mov_avg_frame).mean())
    ylim = find_max_min(temp_list, num_MM)
    for i in range(num_MM):
        plt.subplot(1, num_MM, i+1)  
        plt.plot(temp_list[i])
        plt.title(f"MM{i+1}_Reward{mov_avg_frame}")
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.axhline(y = 0, color = 'black', linestyle = '-')
        plt.ylim(ylim)
    
    #Average action_size comparison:
    plt.figure()
    temp_list = []
    for i in range(num_MM):
        temp_list.append(data_list[i]["action_size"].rolling(mov_avg_frame).mean()*100)
    ylim = find_max_min(temp_list, num_MM)
    for i in range(num_MM):
        plt.subplot(1, num_MM, i+1)  
        plt.plot(temp_list[i])
        plt.title(f"MM{i+1}_Action_size{mov_avg_frame}")
        plt.xlabel('Step')
        plt.ylabel('%BuyingPower')
        plt.axhline(y = 0, color = 'black', linestyle = '-')
        plt.ylim(ylim)

    #Average action_sym comparison:
    plt.figure()
    temp_list = []
    for i in range(num_MM):
        temp_list.append(data_list[i]["action_sym"].rolling(mov_avg_frame).mean())
    ylim = find_max_min(temp_list, num_MM)
    for i in range(num_MM):
        plt.subplot(1, num_MM, i+1)  
        plt.plot(temp_list[i])
        plt.title(f"MM{i+1}_Action_Sym{mov_avg_frame}")
        plt.xlabel('Step')
        #plt.ylabel('Action')
        plt.axhline(y = 0, color = 'black', linestyle = '-')
        plt.ylim(ylim)

    #Average action_asym comparison:
    plt.figure()
    temp_list = []
    for i in range(num_MM):
        temp_list.append(data_list[i]["action_asym"].rolling(mov_avg_frame).mean())
    ylim = find_max_min(temp_list, num_MM)
    for i in range(num_MM):
        plt.subplot(1, num_MM, i+1)  
        plt.plot(temp_list[i])
        plt.title(f"MM{i+1}_Action_Asym{mov_avg_frame}")
        plt.xlabel('Step')
        #plt.ylabel('Action')
        plt.axhline(y = 0, color = 'black', linestyle = '-')
        plt.ylim(ylim)
        
    #Average Cum_Pnl comparison:
    plt.figure()
    temp_list = []
    for i in range(num_MM):
        temp_list.append(data_list[i]["recent_pl_change"].cumsum())
    ylim = find_max_min(temp_list, num_MM)
    for i in range(num_MM):
        plt.subplot(1, num_MM, i+1)  
        plt.plot(temp_list[i])
        plt.title(f"MM{i+1}_Cum_PnL")
        plt.xlabel('Step')
        plt.ylabel('Returns')
        plt.axhline(y = 0, color = 'black', linestyle = '-')
        plt.ylim(ylim)
        
    #Average change_in_inventory_pnl comparison:
    plt.figure()
    temp_list = []
    for i in range(num_MM):
        temp_list.append(data_list[i]["change_inven_pnl"])
    ylim = find_max_min(temp_list, num_MM)
    for i in range(num_MM):
        plt.subplot(1, num_MM, i+1)  
        plt.plot(temp_list[i])
        plt.title(f"MM{i+1}_change_inven_pnl")
        plt.xlabel('Step')
        plt.axhline(y = 0, color = 'black', linestyle = '-')
        plt.ylim(ylim)
        
    #Average Current_Inventory comparison:
    plt.figure()
    temp_list = []
    for i in range(num_MM):
        temp_list.append(data_list[i]["current_inventory"])
    ylim = find_max_min(temp_list, num_MM)
    for i in range(num_MM):
        plt.subplot(1, num_MM, i+1)  
        plt.plot(temp_list[i])
        plt.title(f"MM{i+1}_Inventory")
        plt.xlabel('Step')
        plt.ylabel('Shares')
        plt.axhline(y = 0, color = 'black', linestyle = '-')
        plt.ylim(ylim)
        

def get_LT_stats(num_LT, mov_avg_frame, folder_num):

    plt.rcParams["figure.figsize"] = [12, 3.50]
    plt.rcParams["figure.autolayout"] = True

    add = 0
    if folder_num == 2:
        add = 10

    data_list = []
    for i in range(num_LT):
        data_list.append(pd.read_csv(f"sep_trader_lt{i+1+add}.csv"))
    
    #Average reward comparison:
    temp_list = []
    for i in range(num_LT):
        temp_list.append(data_list[i]["rew"].rolling(mov_avg_frame).mean())
    ylim = find_max_min(temp_list, num_LT)

    num_rows = math.ceil(num_LT/4)
    for row in range(num_rows):
        plt.figure()

        if row == num_rows-1:#check for last row for uneven division
            num_plt_this_row = num_LT % 4
        else: num_plt_this_row = 4

        for i in range(num_plt_this_row):
            plt.subplot(1, num_MM, i+1) 
            xth_agent = i+(row*4)
            #temp = data_list[xth_agent]["rew"].rolling(mov_avg_frame).mean()
            plt.plot(temp_list[xth_agent])
            plt.title(f"LT{i+1}_Reward{mov_avg_frame}")
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.axhline(y = 0, color = 'black', linestyle = '-')
            plt.ylim(ylim)

    #Average action comparison:
    temp_list = []
    for i in range(num_LT):
        temp_list.append(data_list[i]["act_dir"].rolling(mov_avg_frame).mean())
    ylim = find_max_min(temp_list, num_LT)

    for row in range(num_rows):
        plt.figure()

        if row == num_rows-1:#check for last row for uneven division
            num_plt_this_row = num_LT % 4
        else: num_plt_this_row = 4

        for i in range(num_plt_this_row):
            plt.subplot(1, num_MM, i+1) 
            xth_agent = i+(row*4)
            #temp = data_list[xth_agent]["act_dir"].rolling(mov_avg_frame).mean()
            plt.plot(temp_list[xth_agent])
            plt.title(f"LT{i+1}_Action{mov_avg_frame}")
            plt.xlabel('Step')
            plt.ylabel('Buy/Sell/Hold')
            plt.axhline(y = 0, color = 'black', linestyle = '-')
            plt.ylim(ylim)

    #Cum_PnL comparison:
    temp_list = []
    for i in range(num_LT):
        temp_list.append(data_list[i]["pnl"].cumsum())
    ylim = find_max_min(temp_list, num_LT)

    for row in range(num_rows):
        plt.figure()

        if row == num_rows-1:#check for last row for uneven division
            num_plt_this_row = num_LT % 4
        else: num_plt_this_row = 4

        for i in range(num_plt_this_row):
            plt.subplot(1, num_MM, i+1) 
            xth_agent = i+(row*4)
            #temp = data_list[xth_agent]["pnl"].cumsum()
            plt.plot(temp_list[xth_agent])
            plt.title(f"LT{i+1}_Cum_PnL")
            plt.xlabel('Step')
            plt.ylabel('Return')
            plt.axhline(y = 0, color = 'black', linestyle = '-')
            plt.ylim(ylim)

    #Current Inventory comparison:
    temp_list = []
    for i in range(num_LT):
        temp_list.append(data_list[i]["inv"].cumsum())
    ylim = find_max_min(temp_list, num_LT)

    for row in range(num_rows):
        plt.figure()

        if row == num_rows-1:#check for last row for uneven division
            num_plt_this_row = num_LT % 4
        else: num_plt_this_row = 4

        for i in range(num_plt_this_row):
            plt.subplot(1, num_MM, i+1) 
            xth_agent = i+(row*4)
            #temp = data_list[xth_agent]["inv"]
            plt.plot(temp_list[xth_agent])
            plt.title(f"LT{i+1}_Inventory")
            plt.xlabel('Step')
            plt.ylabel('Shares')
            plt.axhline(y = 0, color = 'black', linestyle = '-')
            plt.ylim(ylim)
            
def generate_report(num_MM, num_LT, mov_avg_frame, output_file, folder_num):
    get_MM_stats(num_MM, mov_avg_frame,folder_num)
    get_LT_stats(num_LT, mov_avg_frame,folder_num)
    __save_image(output_file)


if __name__ == "__main__":
    os.chdir("/home/shiftpub/Results_Simulation/iteration_info")
    num_MM = 4
    num_LT = 10
    num_periods = 1
    mov_avg_frame = 100
    output_file = "_Agent_Stat.pdf"
    generate_report(num_MM, num_LT, mov_avg_frame, output_file, 1)

    # os.chdir("/home/shiftpub/Results_Simulation2/iteration_info")
    # num_MM = 4
    # num_LT = 10
    # num_periods = 1
    # mov_avg_frame = 100
    # output_file = "_Agent_Stat.pdf"
    # generate_report(num_MM, num_LT, mov_avg_frame, output_file, 2)
