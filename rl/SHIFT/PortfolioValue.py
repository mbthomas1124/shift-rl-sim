import pandas as pd
import numpy as np
import time
import shift

#cd /mnt/c/Users/liuyu/PycharmProjects/FE800/SHIFT

trader = shift.Trader("test003")
trader.connect("initiator.cfg", "password")
trader.isConnected()
trader.disconnect()
#-------------------------------------
trader.subAllOrderBook()
trader.getLastPrice("AAPL")
trader.getClosePrice("AAPL", True, 1)
trader.getBestPrice("AAPL").getBidPrice()
trader.getBestPrice("AAPL").getAskPrice()
trader.getBestPrice("AAPL").getAskSize()

trader.getPortfolioSummary().getTotalBP()

print("Symbol\t\tShares\t\tPrice\t\tP&L\t\tTimestamp")
for item in trader.getPortfolioItems().values():
    print("%6s\t\t%6d\t%9.2f\t%7.2f\t\t%26s" %
          (item.getSymbol(), item.getShares(), item.getPrice(), item.getRealizedPL(), item.getTimestamp()))

print("Symbol\t\t\t\t\t Type\t  Price\t\tSize\tID\t\t\t\t\t\t\t\t\t\tTimestamp")
for order in trader.getWaitingList():
    print("%6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s" %
          (order.symbol, order.type, order.price, order.size, order.id, order.timestamp))

print("Symbol\t\t\t\t\t Type\t  Price\t\tSize\tID\t\t\t\t\t\t\t\t\t\tTimestamp")
for order in trader.getSubmittedOrders():
    print("%6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s" %
          (order.symbol, order.type, order.price, order.size, order.id, order.timestamp))
trader.getSubmittedOrdersSize()
#--------------------------------------------------------------------------
#Place Order
trader.submitOrder(shift.Order(shift.Order.LIMIT_BUY, "AAPL", 5, 172.81))
trader.submitOrder(shift.Order(shift.Order.LIMIT_SELL, "AAPL", 5, 90.00))
#--------------------------------------------------------------------------
trader.cancelAllPendingOrders()
trader.getPortfolioSummary().getTotalRealizedPL()



def portfolioItems():
    pi = trader.getPortfolioItems().values()
    idx = 0
    df_pi = pd.DataFrame(columns=['Symbol', 'Share', 'Close_Price'])
    for order in pi:
        df_pi.loc[idx, ['Symbol', 'Share']] = [order.getSymbol(), order.getShares()]
        idx += 1
        df_pi = df_pi[df_pi['Share'] != 0]

        df_pi.index = np.arange(0, len(df_pi))
    for i in range(0, len(df_pi.index)):
        df_pi.loc[i, ['Close_Price']] = trader.getClosePrice(df_pi["Symbol"][i], df_pi["Share"][i] < 0,
                                                             int(abs(df_pi["Share"][i] / 100)))
    return df_pi


def waitingList():
    wl = trader.getWaitingList()
    idx = 0
    df_wl = pd.DataFrame(columns=['Size', 'Price', 'Type'])
    for order in wl:
        df_wl.loc[idx, ['Size', 'Price', 'Type']] = [order.size, order.price, order.type]
        idx += 1
        # df_wl = df_wl[df_wl['Type'] == shift.Order.OrderType.LIMIT_BUY]
    return df_wl

def portfolioValue():
    ps = trader.getPortfolioSummary()
    portfolio = portfolioItems()
    waiting = waitingList()
    wait_buy = waiting[waiting['Type'] == shift.Order.OrderType.LIMIT_BUY]
    return sum((wait_buy.Price * wait_buy.Size * 100)) + ps.getTotalBP() + sum((portfolio.Close_Price * portfolio.Share))

trader.getPortfolioSummary().getTotalBP()
# def reward():
#     # next_obs = self._get_obs()
#     p_v = portfolioValue()
#     print(f'portfolio_value: {p_v}')
#     reward = np.log(p_v / last_value)
#     last_value = p_v
#     return reward


def clearPosition():
    pt = portfolioItems()
    pt.index = np.arange(0, len(pt))
    for i in range(0, len(pt.index)):
        if (pt.loc[i, ['Share']] > 0).bool():
            trader.submitOrder(shift.Order(shift.Order.MARKET_SELL, pt["Symbol"][i], int(abs(pt["Share"][i] / 100)), 0))
        else:
            trader.submitOrder(shift.Order(shift.Order.MARKET_BUY, pt["Symbol"][i], int(abs(pt["Share"][i] / 100)), 0))

    trader.cancelAllPendingOrders()
    print('Position Cleared.')




#------------------------------------------------------------------------------
#PV 1st draft
def PortfolioValue1(): #Using Close Price to calculate holding stocks' value
    idx = 0
    df_pi = pd.DataFrame(columns=['Symbol', 'Share', 'Close_Price'])
    df_wl = pd.DataFrame(columns=['Size', 'Price', 'Type'])
    pi = trader.getPortfolioItems().values()
    wl = trader.getWaitingList()
    ps = trader.getPortfolioSummary()

    assert pi
    for order in pi:
        df_pi.loc[idx, ['Symbol', 'Share']] = [order.getSymbol(), order.getShares()]
        idx += 1
        df_pi = df_pi[df_pi['Share'] != 0]
        df_pi.index = np.arange(0, len(df_pi))
    for i in range(0, len(df_pi.index)):
        df_pi.loc[i, ['Close_Price']] = trader.getClosePrice(df_pi["Symbol"][i], df_pi["Share"][i] < 0,
                                                             int(abs(df_pi["Share"][i] / 100)))

    assert wl
    for order in wl:
        df_wl.loc[idx, ['Size', 'Price', 'Type']] = [order.size, order.price, order.type]
        idx += 1
        df_wl = df_wl[df_wl['Type'] == shift.Order.OrderType.LIMIT_BUY]

    pv = sum((df_wl.Price * df_wl.Size * 100)) + ps.getTotalBP() + sum((df_pi.Close_Price * df_pi.Share))
    print(pv)


def PortfolioValue2():
    idx = 0
    df_pi = pd.DataFrame(columns = ['Symbol', 'Share', 'Best_Price'])
    df_wl = pd.DataFrame(columns = ['Size', 'Price', 'Type'])
    pi = trader.getPortfolioItems().values()
    wl = trader.getWaitingList()
    ps = trader.getPortfolioSummary()

    assert pi
    for order in pi:
        df_pi.loc[idx, ['Symbol', 'Share']] = [order.getSymbol(), order.getShares()]
        idx += 1
        df_pi = df_pi[df_pi['Share']!=0]
        df_pi.index = np.arange(0, len(df_pi))
    for i in range(0, len(df_pi.index)):
        if (df_pi.loc[i, ['Share']] > 0).bool():
            df_pi.loc[i, ['Best_Price']] = trader.getBestPrice(df_pi['Symbol'][i]).getBidPrice()
        else:
            df_pi.loc[i, ['Best_Price']] = trader.getBestPrice(df_pi['Symbol'][i]).getAskPrice()

    assert wl
    for order in wl:
        df_wl.loc[idx, ['Size', 'Price', 'Type']] = [order.size, order.price, order.type]
        idx += 1
    df_wl=df_wl[df_wl['Type']==shift.Order.OrderType.LIMIT_BUY]

    pv = sum((df_wl.Price * df_wl.Size*100)) + ps.getTotalBP() + sum((df_pi.Best_Price * df_pi.Share))
    print(pv)
#-------------------------------------------------------------------------------------------------------



if __name__=='__main__':

    trader = shift.Trader("test004")
    trader.disconnect()
    trader.connect("initiator.cfg", "password")
    trader.subAllOrderBook()

    trader.submitOrder(shift.Order(shift.Order.MARKET_BUY, "AAPL", 1, 0.00))
    trader.submitOrder(shift.Order(shift.Order.MARKET_SELL, "IBM", 1, 0.00))
    trader.submitOrder(shift.Order(shift.Order.MARKET_SELL, "AAPL", 1, 0.00))
    trader.submitOrder(shift.Order(shift.Order.MARKET_BUY, "AXP", 1, 0.00))
    trader.submitOrder(shift.Order(shift.Order.LIMIT_BUY, "IBM", 1, 170.00))
    trader.submitOrder(shift.Order(shift.Order.LIMIT_SELL, "AXP", 1, 100.00))

    time.sleep(1)
    portfolioItems()
    waitingList()
    print(portfolioValue())

    trader.disconnect()