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

def flash_crash(trader, direction, total_size, num_orders, sleep_t,  num_flash_crash = 1, crash_sleep = 30):
    best_p = trader.get_best_price("CS1")
    sp = round((best_p.get_ask_price() - best_p.get_bid_price()),4)
    mid = round(((best_p.get_ask_price()+best_p.get_bid_price())/2),4)
    
    size = int(total_size / num_orders)
    #sleep(crash_sleep)
    for crash in range(num_flash_crash):
        if direction == "buy":
            for i in range(num_orders):
                market_buy = shift.Order(shift.Order.Type.MARKET_BUY, "CS1", size)
                trader.submit_order(market_buy)
                sleep(sleep_t)
        if direction == "sell":
            for i in range(num_orders):
                market_sell = shift.Order(shift.Order.Type.MARKET_SELL, "CS1", size)
                trader.submit_order(market_sell)
                sleep(sleep_t)
        sleep(crash_sleep)
        print("Realized Pnl", trader.get_portfolio_summary().get_total_realized_pl())


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
        sleep(5)
        #flash_crash(trader, direction, total_size, num_orders, sleep_t,  num_flash_crash = 1, crash_sleep = 30)
        flash_crash(trader1, "sell", 2000, 10, 1, 5, 120)
    except KeyboardInterrupt:
        trader1.disconnect()
    finally:
        trader1.disconnect()
