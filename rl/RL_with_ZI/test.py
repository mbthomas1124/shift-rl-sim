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

def LOB_record(trader, trail):
    check_frequency = 1
    print("START")

    num_tickers = 1

    tickers = []
    for i in range(num_tickers):
        tickers.append({"sp":[], "mid":[], "ask_5":[],"ask_4":[],"ask_3":[],"ask_2":[],"ask_1":[], "bid_1":[], "bid_2":[], "bid_3":[], "bid_4":[], "bid_5":[]})
    try:
        while True:
            #get price
            for i in range(num_tickers):
                tick = 'CS'+str(i+1)
                best_p = trader.get_best_price(tick)
                sp = round((best_p.get_ask_price() - best_p.get_bid_price()),4)
                mid = round(((best_p.get_ask_price()+best_p.get_bid_price())/2),4)
                tickers[i]["sp"].append(sp)
                tickers[i]["mid"].append(mid)
                ask_book = trader.get_order_book(tick, shift.OrderBookType.LOCAL_ASK, 5)
                bid_book = trader.get_order_book(tick, shift.OrderBookType.LOCAL_BID, 5)
                # get data
                bid_size = []
                ask_size = []
                for order in ask_book:
                    ask_size.append(order.size) 
                for order in bid_book:
                    bid_size.append(order.size)

                while len(ask_size) < 5:
                    ask_size.append(0)
                while len(bid_size) < 5:
                    bid_size.append(0)
                #print("bid ask volume:", bid_size_1, ask_size_1)

                tickers[i]["ask_1"].append(ask_size[0])
                tickers[i]["ask_2"].append(ask_size[1])
                tickers[i]["ask_3"].append(ask_size[2])
                tickers[i]["ask_4"].append(ask_size[3])
                tickers[i]["ask_5"].append(ask_size[4])
                tickers[i]["bid_1"].append(bid_size[0])
                tickers[i]["bid_2"].append(bid_size[1])
                tickers[i]["bid_3"].append(bid_size[2])
                tickers[i]["bid_4"].append(bid_size[3])
                tickers[i]["bid_5"].append(bid_size[4])
                
            sleep(check_frequency)
    except:
        for i in range(num_tickers):
            file_path = f"./iteration_info/Orderbook_trial{trail}_CS{i+1}.csv"
            pd.DataFrame(tickers[i]).to_csv(file_path, index=False)
        print("Recording Thread DONE")



if __name__ == '__main__':
    trader1 = shift.Trader("test001")

    try:
        trader1.connect("initiator2.cfg", "password")
        #subscribe order book for all tickers
        trader1.sub_all_order_book()
        
    except shift.IncorrectPasswordError as e:
        print(e)
    except shift.ConnectionTimeoutError as e:
        print(e)
    try:
        sleep(5)
        LOB_record(trader1,"AMM")
    except KeyboardInterrupt:
        trader1.disconnect()
    finally:
        trader1.disconnect()
