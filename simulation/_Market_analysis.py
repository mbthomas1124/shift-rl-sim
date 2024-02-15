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
from scipy.stats import skew, kurtosis
import numpy as np
import os
import powerlaw



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


def __get_series(ob):
    # given a tuple of lists of lists of order book entries, returns midprice list and spread list
    bid_ob, ask_ob = ob
    midprices = []
    spreads = []
    for i in range(len(bid_ob)):
        best_bid = None
        best_ask = None
        if len(bid_ob[i]) > 0:
            best_bid = bid_ob[i][0].price
        if len(ask_ob[i]) > 0:
            best_ask = ask_ob[i][0].price

        if (best_bid == None) and (best_ask == None):
            raise Exception("both sides of the order book are empty")
        if best_bid == None:
            midprices.append(best_ask)
            spreads.append(0)
        elif best_ask == None:
            midprices.append(best_bid)
            spreads.append(0)
        else:
            midprices.append((best_ask + best_bid) / 2)
            spreads.append(best_ask - best_bid)
    return midprices, spreads


def get_stats(in_data, per_num, trade_recs):
    # PARAMS:
    # ob: a tuple containing a list of bid-side order books and a list of spread-side order books

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # bid_ob, ask_ob = ob
    # mp, sp = __get_series(ob)
    trade_recs = trade_recs[trade_recs["decision"] == "2"]
    trade_recs = trade_recs[trade_recs["destination"] == "SHIFT"]
    trade_recs['execution_time'] = pd.to_datetime(trade_recs['execution_time'])
    trade_recs = trade_recs.set_index("execution_time")
    trade_recs["sum_over_time"] = trade_recs["size"].rolling("5s").sum()


    data = pd.DataFrame()

    data["midprice"] = in_data["curr_mp"]
    data["spread"] = in_data["spread"]

    return_timesteps = 4
    data["returns"] = data["midprice"].pct_change(1)

    data["bid_volume"] = np.sum(
        [
            in_data["bid_1"],
            in_data["bid_2"],
            in_data["bid_3"],
            in_data["bid_4"],
            in_data["bid_5"],
        ],
        axis=0,
    )
    data["ask_volume"] = np.sum(
        [
            in_data["ask_1"],
            in_data["ask_2"],
            in_data["ask_3"],
            in_data["ask_4"],
            in_data["ask_5"],
        ],
        axis=0,
    )
    volumes = np.mean(
        [
            in_data["bid_5"],
            in_data["bid_4"],
            in_data["bid_3"],
            in_data["bid_2"],
            in_data["bid_1"],
            in_data["ask_1"],
            in_data["ask_2"],
            in_data["ask_3"],
            in_data["ask_4"],
            in_data["ask_5"],
        ],
        axis=1,
    )
    data["volume_inbalance"] = np.subtract(data["bid_volume"], data["ask_volume"])

    with pd.option_context("mode.use_inf_as_null", True):
        data = data.dropna()

    # midprice line plot
    plt.figure()
    plt.plot(data["midprice"])
    plt.title(f"Period{per_num}_Midprice")
    print("1")

    # returns line plot
    plt.figure()
    plt.plot(data["returns"])
    plt.axhline(y=0, color="r", linestyle="-")
    plt.title(f"Period{per_num}_Returns")
    print("2")

    #returns histrogram
    # plt.figure()
    # ax = sns.histplot(data=data, x="returns", label="returns")
    # map_pdf(data["returns"], ax)
    # ax.legend()
    # plt.title(f"Period{per_num}_Returns Distribution")
    # print("SSS")

    # returns qqplot
    plt.figure()
    sm.qqplot(data["returns"], fit=True, line="45")
    plt.title(f"Period{per_num}_Returns QQ Plot")

    # returns autocorrelation
    plt.figure()
    statsmodels.graphics.tsaplots.plot_acf(
        x=data["returns"], lags=np.arange(1, 11), alpha=0.05, auto_ylims=True
    )
    plt.title(f"Period{per_num}_Returns Autocorrelation")

    #returns squared autocorrelation
    plt.figure()
    statsmodels.graphics.tsaplots.plot_acf(
        x=(data["returns"] ** 2), lags=np.arange(1, 11), alpha=0.05, auto_ylims=True
    )
    plt.title(f"Period{per_num}_Returns^2 Autocorrelation")

    # spread line plot
    plt.figure()
    plt.plot(data["spread"])
    plt.title(f"Period{per_num}_Spread")
    print("3")
    #spread histogram
    # plt.figure()
    # sns.histplot(data=data, x="spread")
    # plt.title(f"Period{per_num}_Spread Distribution")
    # print("4")
    #spread autocorrelation
    plt.figure()
    statsmodels.graphics.tsaplots.plot_acf(
        x=(data["spread"]), lags=np.arange(1, 11), alpha=0.05, auto_ylims=True
    )
    plt.title(f"Period{per_num}_Spread Autocorrelation")
    print("5")
    # volume inbalance
    plt.figure()
    plt.plot(data["volume_inbalance"])
    plt.axhline(y=0, color="r", linestyle="-")
    plt.title(f"Period{per_num}_volume_inbalance")
    print("6")
    # ob volumes
    plt.figure()
    ax = plt.plot(volumes, "ro")
    plt.xlabel("Bid ----- Ask")
    plt.axvline(x=4.5, color="r", label="midprice")
    plt.legend()
    plt.title(f"Period{per_num}_order book volumes")

     # trade volume
    plt.figure()
    ax = plt.plot(trade_recs["size"])
    plt.title(f"trading volume per transaction, Average:{round(sum(trade_recs['size']/len(trade_recs['size'])),4)}")


    # trade volume summed over time
    plt.figure()
    ax = plt.plot(trade_recs["sum_over_time"])
    plt.title(f"trading volume per 5 secs, Average per 5s:{round(sum(trade_recs['size'])/36000*5, 4)}")

    print(f"Skewness:{skew(volumes)}, Kurtosis:{kurtosis(data['returns'])}")

    print("alpha:", powerlaw.Fit(volumes[5:]).power_law.alpha)

def generate_report(data, num_per, filename, tr, manualy_selection = None):
    df = data
    get_stats(df, " whole", tr)
    if num_per != 1:
        split_data = np.array_split(df, num_per)
        for i in range(num_per):
            get_stats(split_data[i], i+1)

    # call the function
    __save_image(filename)


if __name__ == "__main__":
    os.chdir("/home/shiftpub/Results_Simulation/iteration_info")
    file = "sep_trader_mm1.csv"
    num_periods = 1
    trading_records = "trade_rec.csv"
    df = pd.read_csv(file)
    tr = pd.read_csv(trading_records)
    generate_report(df, num_periods, "_Market_stat.pdf", tr)

    # file = "AMZN_LOB.csv"
    # num_periods = 1
    # df = pd.read_csv(file)
    # generate_report(df, num_periods, "_AMZN_Market_stat.pdf")

    # file = "INTC_LOB.csv"
    # num_periods = 1
    # df = pd.read_csv(file)
    # generate_report(df, num_periods, "_INTC_Market_stat.pdf")

   

    # os.chdir("/home/shiftpub/Results_Simulation2/iteration_info")
    # file = "sep_trader_mm5.csv"
    # num_periods = 3
    # df = pd.read_csv(file)
    # generate_report(df, num_periods, "_Market_stat.pdf")