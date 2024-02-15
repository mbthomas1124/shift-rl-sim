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

def flash_crash(trader, direction, total_size, num_orders, sleep_t,  num_flash_crash = 1, crash_sleep = 30, ticker = "CS1"):
    order_cols = ["timestep", "Type", "Status", "Symbol", "Size", "Executed_size", "Price", "Executed Price", "ID"]
    order_df = pd.DataFrame(columns = order_cols)
    size = int(total_size / num_orders)
    #sleep(crash_sleep)
    for crash in range(num_flash_crash):
        sleep(crash_sleep)#

        # if direction == "buy":
        #     for i in range(num_orders):
        #         market_buy = shift.Order(shift.Order.Type.MARKET_BUY, "CS1", size)
        #         trader.submit_order(market_buy)
        #         sleep(sleep_t)
        last_available_price = 0
        last_available_sp = 0.01
        for i in range(num_orders):
            best_p = trader.get_best_price(ticker)
            sp = round((best_p.get_ask_price() - best_p.get_bid_price()),4)
            mid = round(((best_p.get_ask_price()+best_p.get_bid_price())/2),4)
            if mid == 0 or min(best_p.get_ask_price(),best_p.get_bid_price()) == 0:
                mid = max(last_available_price, max(best_p.get_ask_price(),best_p.get_bid_price())) 
                sp = last_available_sp
            last_available_price = mid
            last_available_sp = sp

            if direction == "sell":
                order_rice = round(mid - 3*sp,2)
                limit_sell = shift.Order(shift.Order.Type.LIMIT_SELL, "CS1", size, order_rice)
                trader.submit_order(limit_sell)
            elif direction == "buy":
                order_rice = round(mid + 3*sp,2)
                limit_buy = shift.Order(shift.Order.Type.LIMIT_BUY, "CS1", size, order_rice)
                trader.submit_order(limit_buy)
            sleep(sleep_t)
        print("Crash",crash+1)
    print("saving")
    #saving the orders submitted
    step = 0
    for order in trader.get_submitted_orders():
        order_info = [order.timestamp, order.type, order.status, order.symbol,order.size, order.executed_size, order.price,
            order.executed_price, order.id]
        order_df.loc[step] = order_info
        step += 1
    order_df.to_csv('/home/shiftpub/Results_Simulation/iteration_info/flash_orders.csv')

if __name__ == '__main__':
    trader1 = shift.Trader("flash_crash_maker")

    try:
        trader1.connect("/home/shiftpub/initiator.cfg", "password")
        #subscribe order book for all tickers
        trader1.sub_all_order_book()
        
    except shift.IncorrectPasswordError as e:
        print(e)
    except shift.ConnectionTimeoutError as e:
        print(e)
    try:
        #flash_crash(trader, direction, total_size, num_orders, sleep_t,  num_flash_crash = 1, crash_sleep = 30, ticker)
        flash_crash(trader1, "sell", 3000, 10, 1, 299, 120, "CS1")
    except KeyboardInterrupt:
        trader1.disconnect()
    finally:
        trader1.disconnect()
