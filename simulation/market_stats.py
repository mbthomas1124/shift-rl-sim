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


def get_stats(symbol, in_data, trade_recs, filename):
    # PARAMS:
    # ob: a tuple containing a list of bid-side order books and a list of spread-side order books
    # filename: the name of the resulting pdf file

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

    trade_recs = trade_recs[trade_recs["symbol"] == symbol]
    trade_recs = trade_recs[trade_recs["decision"] == 2]
    trade_recs = trade_recs[trade_recs["destination"] == "SHIFT"]
    trade_recs['execution_time'] = pd.to_datetime(trade_recs['execution_time'])
    trade_recs = trade_recs.set_index("execution_time")
    trade_recs["sum_over_time"] = trade_recs["size"].rolling("2s").sum()

    with pd.option_context("mode.use_inf_as_null", True):
        data = data.dropna()

    # midprice line plot
    plt.figure()
    plt.plot(data["midprice"])
    plt.title("Midprice")

    # returns line plot
    plt.figure()
    plt.plot(data["returns"])
    plt.axhline(y=0, color="r", linestyle="-")
    plt.title("Returns")

    # returns histrogram
    plt.figure()
    ax = sns.histplot(data=data, x="returns", label="returns")
    map_pdf(data["returns"], ax)
    ax.legend()
    plt.title("Returns Distribution")

    # returns qqplot
    plt.figure()
    sm.qqplot(data["returns"], fit=True, line="45")
    plt.title("Returns QQ Plot")

    # returns autocorrelation
    plt.figure()
    statsmodels.graphics.tsaplots.plot_acf(
        x=data["returns"], lags=np.arange(1, 11), alpha=0.05, auto_ylims=True
    )
    plt.title("Returns Autocorrelation")

    # returns squared autocorrelation
    plt.figure()
    statsmodels.graphics.tsaplots.plot_acf(
        x=(data["returns"] ** 2), lags=np.arange(1, 11), alpha=0.05, auto_ylims=True
    )
    plt.title("Returns^2 Autocorrelation")

    # spread line plot
    plt.figure()
    plt.plot(data["spread"])
    plt.title("Spread")

    # spread histogram
    plt.figure()
    sns.histplot(data=data, x="spread")
    plt.title("Spread Distribution")

    # spread autocorrelation
    plt.figure()
    statsmodels.graphics.tsaplots.plot_acf(
        x=(data["spread"]), lags=np.arange(1, 11), alpha=0.05, auto_ylims=True
    )
    plt.title("Spread Autocorrelation")

    # volume inbalance
    plt.figure()
    plt.plot(data["volume_inbalance"])
    plt.axhline(y=0, color="r", linestyle="-")
    plt.title("volume_inbalance")

    # ob volumes
    plt.figure()
    ax = plt.plot(volumes, "ro")
    plt.xlabel("Bid ----- Ask")
    plt.axvline(x=4.5, color="r", label="midprice")
    plt.legend()
    plt.title("order book volumes")

    # trade volume
    plt.figure()
    ax = plt.plot(trade_recs["size"])
    plt.title("trading volume per transaction")


    # trade volume summed over time
    plt.figure()
    ax = plt.plot(trade_recs["sum_over_time"])
    plt.title("trading volume per 2 secs")

    # call the function
    __save_image(filename)


if __name__ == "__main__":
    ticker = "CS1"
    file = "sep_trader_mm1(3).csv"
    trading_records = "trading_records.csv"
    df = pd.read_csv(file)
    tr = pd.read_csv(trading_records)
    get_stats(ticker, df, tr, "stats_3.pdf")
