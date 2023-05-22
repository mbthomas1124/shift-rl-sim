from threading import Thread
import shift
import datetime as dt
import pandas as pd
from time import sleep


def thread_test(trader, i, duration, results):
    ticker = "CS1"
    check_freq = 0.4

    mid_p = []
    current_time = []

    previous_pl = trader.get_portfolio_summary().get_total_realized_pl()

    end = trader.get_last_trade_time() + dt.timedelta(minutes=duration)
    print(trader.get_last_trade_time())
    print(end)
    while trader.get_last_trade_time() < end:
        # cancel all the remaining orders
        for order in trader.get_waiting_list():
            if (order.symbol == ticker):
                trader.submit_cancellation(order)
                sleep(0.2)

        best_p = trader.get_best_price(ticker)
        mid = round(((best_p.get_ask_price()+best_p.get_bid_price())/2),4)
        mid_p.append(mid)
        current_time.append(trader.get_last_trade_time())
        order_s = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, 1, best_p.get_ask_price())
        order_b = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, 1, best_p.get_bid_price())
        trader.submit_order(order_s)
        trader.submit_order(order_b)
        #print(trader.get_last_trade_time())
        #print(f"test00{i}   {trader.get_last_trade_time()}   {end}")
        sleep(check_freq)
    

    results[f'test00{i}_midprice'] = pd.Series(mid_p)
    results[f'test00{i}_last_trade_time'] = pd.Series(current_time)
    print(f'test00{i}total bp:',trader.get_portfolio_summary().get_total_bp())
    print(f'test00{i}total PnL:',trader.get_portfolio_summary().get_total_realized_pl() - previous_pl)

if __name__ == "__main__":
    threads = []
    results = {}
    duration = 1
    traders = []
    num_tra = 1
    for i in range(num_tra):
        traders.append(shift.Trader(f"test00{i+1}"))
        try:
            traders[i].connect("initiator.cfg", "password")
            #subscribe order book for all tickers
            traders[i].sub_all_order_book()
        except shift.IncorrectPasswordError as e:
            print(e)
        except shift.ConnectionTimeoutError as e:
            print(e)
    sleep(5)
    print(traders)
    for i in range(num_tra):
        threads.append(Thread(target=thread_test, args =(traders[i], i+1, duration, results)))

    try:
        for thread in threads:        
            thread.start()
            sleep(1)
        
        #sleep((duration+1) * 60)

        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        for i in range(num_tra):
            traders[i].disconnect()
    finally:
        for i in range(num_tra):
            traders[i].disconnect()

    #print(results)   
    df = pd.DataFrame.from_dict(results).to_csv("results.csv", index = False)
    
        

