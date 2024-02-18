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
from threading import Thread
from test_flash_crash import flash_crash


def LOB_record(trader, hours):
    check_frequency = 1

    print("START")

    num_tickers = 1
    step = 0
    total_step = hours * 3600

    tickers = []
    for i in range(num_tickers):
        tickers.append({"time":[], "sp":[], "mid":[], "Bid_depth":[], "Ask_depth":[], "ask_5":[],"ask_5_p":[], "ask_4":[],"ask_4_p":[],"ask_3":[],"ask_3_p":[],"ask_2":[],"ask_2_p":[],"ask_1":[],"ask_1_p":[],
                                                                            "bid_1_p":[],"bid_1":[], "bid_2_p":[],"bid_2":[], "bid_3_p":[],"bid_3":[], "bid_4_p":[],"bid_4":[], "bid_5_p":[],"bid_5":[]})
    try:
        while step < total_step:
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
                bid_p = []
                ask_p = []
                for order in ask_book:
                    ask_size.append(order.size) 
                    ask_p.append(order.price) 
                for order in bid_book:
                    bid_size.append(order.size)
                    bid_p.append(order.price) 

                while len(ask_size) < 5:
                    ask_size.append(0)
                    ask_p.append(0)
                while len(bid_size) < 5:
                    bid_size.append(0)
                    bid_p.append(0)

                ask_book = trader.get_order_book(tick, shift.OrderBookType.LOCAL_ASK, 5)
                bid_book = trader.get_order_book(tick, shift.OrderBookType.LOCAL_BID, 5)
                bid_depth = int(len(bid_book))
                ask_depth = int(len(ask_book))
                tickers[i]["time"].append([datetime.now()])
                #print("bid ask volume:", bid_size_1, ask_size_1)
                tickers[i]["Bid_depth"].append(bid_depth)
                tickers[i]["Ask_depth"].append(ask_depth)

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

                tickers[i]["ask_1_p"].append(ask_p[0])
                tickers[i]["ask_2_p"].append(ask_p[1])
                tickers[i]["ask_3_p"].append(ask_p[2])
                tickers[i]["ask_4_p"].append(ask_p[3])
                tickers[i]["ask_5_p"].append(ask_p[4])
                tickers[i]["bid_1_p"].append(bid_p[0])
                tickers[i]["bid_2_p"].append(bid_p[1])
                tickers[i]["bid_3_p"].append(bid_p[2])
                tickers[i]["bid_4_p"].append(bid_p[3])
                tickers[i]["bid_5_p"].append(bid_p[4])
                
            sleep(check_frequency)
            step+=1
    except:
        for i in range(num_tickers):
            file_path = f".ZI_CS{i+1}.csv"
            pd.DataFrame(tickers[i]).to_csv(file_path, index=False)
        print("Recording Thread DONE")
    finally:
        for i in range(num_tickers):
            file_path = f".ZI_CS{i+1}.csv"
            pd.DataFrame(tickers[i]).to_csv(file_path, index=False)
        print("Recording Thread DONE")



if __name__ == '__main__':
    trader1 = shift.Trader("test001")

    try:
        trader1.connect("initiator.cfg", "password")
        #subscribe order book for all tickers
        trader1.sub_all_order_book()
        sleep(1)
        

        #connect flash crash agent
        crash_maker = shift.Trader("flash_crash_maker_01")
        crash_maker.disconnect()
        crash_maker.connect("/home/shiftpub/initiator.cfg", "password")
        crash_maker.sub_all_order_book()

        flash_config = {"buy/sell": "buy",
                         "flash_size": 1500,
                         "num_orders": 5,
                         "time_bet_order": 1,
                         "num_flash":88,
                         "time_bet_flash": 400}
        flash_crash_thread = Thread(target=flash_crash, args=(crash_maker,  flash_config["buy/sell"], flash_config["flash_size"], flash_config["num_orders"], flash_config["time_bet_order"],
                                                            flash_config["num_flash"], flash_config["time_bet_flash"], "CS1"))
        lob_rec_thread = Thread(target=LOB_record, args=(trader1, 9.5))

        flash_crash_thread.start()
        lob_rec_thread.start()
        flash_crash_thread.join()
        lob_rec_thread.join()

    except shift.IncorrectPasswordError as e:
        print(e)
    except shift.ConnectionTimeoutError as e:
        print(e)
    
    except KeyboardInterrupt:
        trader1.disconnect()
    finally:
        trader1.disconnect()
