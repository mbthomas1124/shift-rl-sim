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
import numpy as np
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


def get_stats(in_data, per_num):
    # PARAMS:
    # ob: a tuple containing a list of bid-side order books and a list of spread-side order books

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # bid_ob, ask_ob = ob
    # mp, sp = __get_series(ob)

    data = pd.DataFrame()

    data["midprice"] = in_data["curr_mp"]
    data["spread"] = in_data["spread"]

    return_timesteps = 4
    data["returns"] = data["midprice"].pct_change(5)

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

    # returns line plot
    plt.figure()
    plt.plot(data["returns"])
    plt.axhline(y=0, color="r", linestyle="-")
    plt.title(f"Period{per_num}_Returns")

    # returns histrogram
    plt.figure()
    ax = sns.histplot(data=data, x="returns", label="returns")
    map_pdf(data["returns"], ax)
    ax.legend()
    plt.title(f"Period{per_num}_Returns Distribution")

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

    # returns squared autocorrelation
    plt.figure()
    statsmodels.graphics.tsaplots.plot_acf(
        x=(data["returns"] ** 2), lags=np.arange(1, 11), alpha=0.05, auto_ylims=True
    )
    plt.title(f"Period{per_num}_Returns^2 Autocorrelation")

    # spread line plot
    plt.figure()
    plt.plot(data["spread"])
    plt.title(f"Period{per_num}_Spread")

    # spread histogram
    plt.figure()
    sns.histplot(data=data, x="spread")
    plt.title(f"Period{per_num}_Spread Distribution")

    # spread autocorrelation
    plt.figure()
    statsmodels.graphics.tsaplots.plot_acf(
        x=(data["spread"]), lags=np.arange(1, 11), alpha=0.05, auto_ylims=True
    )
    plt.title(f"Period{per_num}_Spread Autocorrelation")

    # volume inbalance
    plt.figure()
    plt.plot(data["volume_inbalance"])
    plt.axhline(y=0, color="r", linestyle="-")
    plt.title(f"Period{per_num}_volume_inbalance")

    # ob volumes
    plt.figure()
    ax = plt.plot(volumes, "ro")
    plt.xlabel("Bid ----- Ask")
    plt.axvline(x=4.5, color="r", label="midprice")
    plt.legend()
    plt.title(f"Period{per_num}_order book volumes")

def generate_report(data, num_per, filename, manualy_selection = None):
    df = data
    split_data = np.array_split(df, num_per)
    for i in range(num_per):
        get_stats(split_data[i], i+1)

    # call the function
    __save_image(filename)


if __name__ == "__main__":
    os.chdir("/home/shiftpub/Results_Simulation/iteration_info")
    file = "sep_trader_mm1.csv"
    num_periods = 3
    df = pd.read_csv(file)
    generate_report(df, num_periods, "_Market_stat.pdf")