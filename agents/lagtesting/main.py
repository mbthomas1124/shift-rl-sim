import shift
from time import sleep
from datetime import datetime, timedelta, time
from threading import Thread
from samplefunc import trade

def main(trader):

    #TEMP
    check_frequency = 10
    current = trader.get_last_trade_time()
    start_time = current
    end_time = current + timedelta(minutes=20)

    processes = []
    tickers = trader.get_stock_list()
    print(tickers)

    for ticker in tickers:
        processes.append(Thread(target=trade, args=(ticker, trader, end_time)))

    for process in processes:
        process.start()

    # wait until endtime is reached
    while trader.get_last_trade_time() < end_time:


        sleep(check_frequency)

    # wait for all processes to finish
    for process in processes:
        process.join(timeout=1)

if __name__ == '__main__':
    sleep(2)
    trader = shift.Trader("xentec")
    try:
        trader.connect("initiator.cfg", "g9VJPrqm")
        # subscribe order book for all tickers
        trader.sub_all_order_book()
    except shift.IncorrectPasswordError as e:
        print(e)
    except shift.ConnectionTimeoutError as e:
        print(e)

    sleep(5)
    main(trader)
    trader.disconnect()
