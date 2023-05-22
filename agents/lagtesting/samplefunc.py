import shift
from time import sleep

def trade(ticker, trader: shift.Trader, end_time):

    while trader.get_last_trade_time() < end_time:
        #bp = trader.get_portfolio_summary().get_total_bp()
        #price = trader.get_last_price(ticker)
        #print(bp, price)
        size = 1
        print('ordering')
        order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, size)
        trader.submit_order(order)

        sleep(10)
