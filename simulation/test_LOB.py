#from socket import timeout
import shift
from time import sleep
from datetime import datetime, timedelta, time
import pandas as pd
import sys
import os
#import numpy as np
#from typing import Dict, Any, Tuple, List
#import math
import datetime as dt
#from threading import Thread

# action = int(sys.argv[1])
# lam = int(sys.argv[2])
# ordersize = int(sys.argv[3])
# orderlen = int(sys.argv[4])
# trial = int(sys.argv[5])
# testtime = int(sys.argv[6])
# withRL = bool(int(sys.argv[7]))


def main(traders):
    check_frequency = 0.5
    # current = trader.get_last_trade_time()
    # #start_time = datetime.combine(current, dt.time(9,30,0))
    # #start_time = datetime.combine(current, dt.time(10,2,0))
    # start_time = current
    # #end_time = datetime.combine(current, dt.time(9,50,0))
    # #end_time = datetime.combine(current, dt.time(10,7,0))
    # end_time = start_time + timedelta(seconds=5)#minutes

    # while trader.get_last_trade_time() < start_time:
    #     print("Still waiting for market open")
    #     sleep(check_frequency)

    print("START")
    trader = traders[0]
    num_tickers = 1

    cols = ["ask_5","ask_4","ask_3","ask_2","ask_1", "bid_1", "bid_2", "bid_3", "bid_4", "bid_5",
        "ask_5_p","ask_4_p","ask_3_p","ask_2_p","ask_1_p", "bid_1_p", "bid_2_p", "bid_3_p", "bid_4_p", "bid_5_p"]
    df = pd.DataFrame(columns = cols)
    df_idx = 0

    order_cols = ["timestep", "Type", "Status", "Symbol", "Size", "Executed_size", "Price", "Executed Price", "ID"]
    order_df = pd.DataFrame(columns = order_cols)
        

    mins = 1
    steps = int(mins * 60 / check_frequency)
    counter = 0
    for x in range(steps):#while trader.get_last_trade_time() < end_time:
        traders[0].submit_order(shift.Order(shift.Order.Type.MARKET_SELL, "CS1", 10))
        traders[1].submit_order(shift.Order(shift.Order.Type.MARKET_BUY, "CS1", 10))
        #get price
        for i in range(num_tickers):
            tick = 'CS'+str(i+1)
            best_p = trader.get_best_price(tick)
            #sp = round((best_p.get_ask_price() - best_p.get_bid_price()),4)
            mid = round(((best_p.get_ask_price()+best_p.get_bid_price())/2),4)


            ask_book = trader.get_order_book(tick, shift.OrderBookType.LOCAL_ASK)
            bid_book = trader.get_order_book(tick, shift.OrderBookType.LOCAL_BID)
            # get data
            bid_size = []
            bid_p = []
            ask_size = []
            ask_p = []
            for order in ask_book:
                ask_size.append(order.size) 
                ask_p.append(order.price) 
            for order in bid_book:
                bid_size.append(order.size)
                bid_p.append(order.price)

            while len(ask_size) < 5:
                ask_size.append(0)
                ask_p.append("nan") 
            while len(bid_size) < 5:
                bid_size.append(0)
                bid_p.append("nan") 
            #print("bid ask volume:", bid_size_1, ask_size_1)
            df.loc[df_idx] = ask_size[::-1] + bid_size + ask_p[::-1] + bid_p
            df_idx +=1
            sleep(check_frequency)
    

    # save results
    file_path = f"LOB.csv"
    # if os.path.isfile(file_path):
    #     df = pd.read_csv(file_path)
    # else:
    #     df = pd.DataFrame(list())
    # for i in range(num_tickers):
    #     trialnum = ((trial) * num_tickers) + i
    #     if withRL:
    #         # with RL
    #         col = f"RL_{trialnum}"
    #     else:
    #         # without RL
    #         col = f"ZI_{trialnum}"
    #     df[col] = pd.Series(df["mid"])
    df.to_csv(file_path, index=False)
    step = 0
    for order in trader.get_submitted_orders():
        order_info = [order.timestamp, order.type, order.status, order.symbol,order.size, order.executed_size, order.price,
            order.executed_price, order.id]
        order_df.loc[step] = order_info
        step += 1
    order_df.to_csv("Order_Market_orders.csv", index=False)
    print("DONE")


if __name__ == '__main__':
    traders = []
    traders.append(shift.Trader("marketmaker_rl_1"))
    traders.append(shift.Trader("marketmaker_rl_2"))
    try:
        for trader in traders:
            trader.connect("initiator.cfg", "password")
            #subscribe order book for all tickers
            trader.sub_all_order_book()
    except shift.IncorrectPasswordError as e:
        print(e)
    except shift.ConnectionTimeoutError as e:
        print(e)
    try:
        sleep(3)
        main(traders)
    except KeyboardInterrupt:
        traders[0].disconnect()
    finally:
        traders[0].disconnect()