# A program to simulate a high-frequency market using hard-coded agents that
# trades in the SHIFT system
import numpy as np
import shift
from collections import deque
from threading import Thread
import threading
from time import sleep
import math
import pandas as pd


class Simple_Traders_Simulator:
    def __init__(self, trader_type_sizes: list, duration: int):
        """
        trader_type_sizes: list has info on how many traders for each type. (0:mm), (1:momentum), (2:fundamental), (3: ofi), (4: twap)
        duration: number of mins the simulation will last
        """
        self.duration = duration
        self.trader_type_sizes = trader_type_sizes
        self.symbol = 'CS1'
        # parameters: short term duration(s), long term duration(s), trading size(lot)
        self.mm_params = [100, 1, 1, 2]
        self.momentum_params = [5, 10, 5]
        self.Fundamental_params = [100, 1, 5]
        self.OFI_params = [5]
        self.TWAP_params = [5, 10, 10, 5]
        self.traders = []
        for i in range(sum(self.trader_type_sizes)+1):
            self.traders.append(f"agent{i+101:03}")
        self.ofi_len = 20
        self.ob_events = deque(maxlen=self.ofi_len)
        self.best_prices = deque(maxlen=self.ofi_len)
        self.midprices = []
        self.spreads = []
        self.barrier = threading.Barrier(sum(trader_type_sizes)+1)
        # for i in range(sum(self.trader_type_sizes)):
        #   traders.append(shift.Trader.connect(f"agent{i+101}"))

    def start_sim(self):
        threads = []
        # MM traders
        start = 0
        for i in range(start, start + self.trader_type_sizes[0]):
            threads.append(Thread(target=self.__MM, args=(
                [self.traders[i], self.mm_params])))
        start += self.trader_type_sizes[0]

        threads.append(Thread(target=self.__orderFlowStream, args=(
            [self.traders[-1]])))

        # Momentum traders

        for i in range(start, start + self.trader_type_sizes[1]):
            threads.append(Thread(target=self.__Momentum, args=(
                [self.traders[i], self.momentum_params])))
        start += self.trader_type_sizes[1]

        # Fundamental traders
        for i in range(start, start + self.trader_type_sizes[2]):
            threads.append(Thread(target=self.__Fundamental, args=(
                [self.traders[i], self.Fundamental_params])))
        start += self.trader_type_sizes[2]

        # OFI traders
        # for i in range(start, start + self.trader_type_sizes[3]):
        #     threads.append(Thread(target=self.__FlowArbitrage, args=(
        #         [self.traders[i], self.OFI_params])))
        # start += self.trader_type_sizes[2]

        # # TWAP traders
        # start += self.trader_type_sizes[0]
        # for i in range(start, start + self.trader_type_sizes[1]):
        #     threads.append(Thread(target=self.__TWAP, args=(
        #         [self.traders[i], self.TWAP_params])))
        #     pass

        # start simulation
        for thread in threads:
            thread.start()
            sleep(0.1)
        # wait for simulation to end
        for thread in threads:
            thread.join()

        data = pd.DataFrame()
        data["midprice"] = self.midprices
        data["spread"] = self.spreads
        data.to_csv("baseline_results.csv", index=False)

    def __Momentum(self, trader_name, params):
        # parameters: short term duration(s), long term duration(s), trading size(lot)
        trader = shift.Trader(trader_name)
        trader.connect("initiator.cfg", "password")
        self.barrier.wait()
        try:
            trader.sub_order_book(self.symbol)
            short_dur = params[0]
            long_dur = params[1]
            size = params[2]
            midprices = deque(maxlen=long_dur)
            countdown = self.duration * 60
            frequency = 1

            while countdown > 0:
                # stores mid price
                bp = trader.get_best_price(self.symbol)
                midprices.append(self.__calculate_midprice(bp))
                if countdown > ((self.duration * 60) - long_dur):
                    print(countdown - ((self.duration * 60) - long_dur))
                    sleep(frequency)
                    countdown -= frequency
                    continue
                # calculate short and long term average prices, #exp average
                short_index = long_dur - short_dur
                short_price = np.array(midprices)[short_index:].mean()
                long_price = np.array(midprices).mean()
                # make buy/sell decision
                if short_price > long_price:
                    order = shift.Order(
                        shift.Order.Type.MARKET_BUY, self.symbol, size)
                    trader.submit_order(order)
                elif short_price < long_price:
                    order = shift.Order(
                        shift.Order.Type.MARKET_SELL, self.symbol, size)
                    trader.submit_order(order)

                sleep(frequency)
                countdown -= frequency

            self.__cancel_and_close(trader)
        except Exception as e:
            trader.disconnect()
            print(e)

    def __TWAP(self, trader, params):
        # parameters: short term duration(s), long term duration(s), trading size(lot), k trading periods for each order
        short_dur = params[0]
        long_dur = params[1]
        size = params[2]
        k = params[3]
        midPrice = deque(maxlen=long_dur)
        countdown = self.duration * 60
        frequency = 1

        while countdown > 0:
            # stores mid price
            midPrice.append(countdown)
            if countdown > (self.duration * 60 - long_dur):
                sleep(frequency)
                print(countdown)
                countdown -= frequency
                continue
            # calculate short and long term average prices
            short_price = np.array(midPrice)[(long_dur - short_dur):].mean()
            long_price = np.array(midPrice).mean()

            # make decision and split orders in k pieces and submit them every frequency seconds
            if short_price > long_price:
                for i in range(k):
                    print(f"bought {size/k} limit lots")
                    sleep(frequency)
                    midPrice.append(countdown)
                countdown -= frequency * k
            elif short_price < long_price:
                print("sell")
                for i in range(k):
                    print(f"sold {size/k} limit lots")
                    sleep(frequency)
                    midPrice.append(countdown)
                countdown -= frequency * k
            else:
                sleep(frequency)
                countdown -= frequency

        self.__cancel_and_close(trader)

    def __Fundamental(self, trader_name, params):
        trader = shift.Trader(trader_name)
        trader.connect("initiator.cfg", "password")
        self.barrier.wait()
        try:
            trader.sub_order_book(self.symbol)
            countdown = self.duration * 60
            check_freq = 1
            target_price = params[0]
            price_variance = params[1]
            size = params[2]
            if (target_price == None):
                bp = trader.get_best_price(self.symbol)
                target_price = self.__calculate_midprice(bp)
            fundametal_price = np.random.normal(target_price, price_variance)

            while countdown:
                bp = trader.get_best_price(self.symbol)
                mp = self.__calculate_midprice(bp)
                if mp > fundametal_price:
                    order = shift.Order(
                        shift.Order.Type.MARKET_SELL, self.symbol, size)
                    trader.submit_order(order)
                elif mp < fundametal_price:
                    order = shift.Order(
                        shift.Order.Type.MARKET_BUY, self.symbol, size)
                    trader.submit_order(order)
                sleep(check_freq)
                countdown -= check_freq

            self.__cancel_and_close(trader)
        except Exception as e:
            trader.disconnect()
            print(e)

    def __FlowArbitrage(self, trader_name, params):
        trader = shift.Trader(trader_name)
        trader.connect("initiator.cfg", "password")
        self.barrier.wait()
        try:
            countdown = self.duration * 60
            check_freq = 1
            size = params[0]

            while countdown:
                w = 0
                v = 0
                data = list(self.ob_events)
                prices = list(self.best_prices)
                for i in range(1, len(self.ob_events)):
                    if prices[i][0] > prices[i-1][0]:
                        w += math.log(data[i][0])
                    elif prices[i][0] == prices[i-1][0]:
                        w += math.log(data[i][0]) - math.log(data[i-1][0])
                    else:
                        w -= math.log(data[i-1][0])

                    if prices[i][1] > prices[i-1][1]:
                        v += math.log(data[i][1])
                    elif prices[i][1] == prices[i-1][1]:
                        v += math.log(data[i][1]) - math.log(data[i-1][1])
                    else:
                        v -= math.log(data[i-1][1])

                log_ofi = w-v
                if log_ofi > 24:
                    order = shift.Order(
                        shift.Order.Type.MARKET_SELL, self.symbol, size)
                    trader.submit_order(order)
                elif log_ofi < -24:
                    order = shift.Order(
                        shift.Order.Type.MARKET_BUY, self.symbol, size)
                    trader.submit_order(order)

                sleep(check_freq)
                countdown -= check_freq

            self.__cancel_and_close(trader)
        except Exception as e:
            trader.disconnect()
            print(e)

    def __MM(self, trader_name, params):
        trader = shift.Trader(trader_name)
        trader.connect("initiator.cfg", "password")
        self.barrier.wait()
        try:
            trader.sub_order_book(self.symbol)
            countdown = self.duration * 60
            check_freq = 0.1
            inital_price = params[0]
            inital_spread = params[1]
            size = params[2]
            kappa = params[3]
            # initial orders
            order = shift.Order(
                shift.Order.Type.LIMIT_BUY, self.symbol, size, inital_price + (inital_spread/2))
            trader.submit_order(order)
            order = shift.Order(
                shift.Order.Type.LIMIT_SELL, self.symbol, size, inital_price - (inital_spread/2))
            trader.submit_order(order)
            while countdown:
                bp = trader.get_best_price(self.symbol)
                v_bid = bp.get_bid_size()
                v_ask = bp.get_ask_size()
                best_bid = bp.get_bid_price()
                best_ask = bp.get_ask_price()
                spread = best_ask - best_bid
                self.midprices.append(self.__calculate_midprice(bp))

                if ((not (v_bid)) or (not v_ask)):
                    self.spreads.append(0.)
                else:
                    self.spreads.append(best_bid-best_ask)

                if ((not (v_bid)) and (not v_ask)):
                    p_t = 0
                else:
                    p_t = (v_bid - v_ask) / (v_bid + v_ask)

                theta = 0.5*(p_t + 1)

                if np.random.rand() < theta:
                    type = shift.Order.Type.LIMIT_SELL
                    eta = spread * np.exp(p_t/kappa)
                    price = best_bid + 1 + math.floor(eta)
                else:
                    type = shift.Order.Type.LIMIT_BUY
                    eta = spread * np.exp(p_t/kappa)
                    price = best_ask - 1 - math.floor(eta)

                order = shift.Order(type, self.symbol, size, price)
                trader.submit_order(order)

                sleep(check_freq)
                countdown -= check_freq

            self.__cancel_and_close(trader)
        except Exception as e:
            trader.disconnect()
            print(e)

    def __calculate_midprice(self, bp: shift.BestPrice):
        # to be implemented in a seperate thread for data collection
        bid = bp.get_bid_price()
        ask = bp.get_ask_price()
        if (bid and ask):
            return np.mean(bid, ask)
        else:
            return max(bid, ask)

    def __orderFlowStream(self, trader_name: str):
        # to be implemented in a seperate thread for data collection
        # while (main.trader == )
        trader = shift.Trader(trader_name)
        trader.connect("initiator.cfg", "password")
        self.barrier.wait()
        try:
            trader.sub_order_book(self.symbol)

            countdown = self.duration * 60
            check_freq = 0.1

            while countdown:
                bp = trader.get_best_price(self.symbol)
                bidvol = bp.get_bid_size()
                askvol = bp.get_ask_size()
                best_bid = bp.get_bid_price()
                best_ask = bp.get_ask_price()
                if (len(self.ob_events) == 0):
                    self.ob_events.append((bidvol, askvol))
                    self.best_prices.append((best_bid, best_ask))
                else:
                    last = self.ob_events.pop()
                    if (bidvol, askvol) != last:
                        self.ob_events.append(last)
                        self.ob_events.append((bidvol, askvol))
                        self.best_prices.append((best_bid, best_ask))
                    else:
                        self.ob_events.append(last)

                sleep(check_freq)
                countdown -= check_freq
        except Exception as e:
            trader.disconnect()
            print(e)

    def __cancel_and_close(self, trader: shift.Trader):
        trader.cancel_all_pending_orders()
        item = trader.get_portfolio_item(self.symbol)
        while item.get_long_shares():
            order = shift.Order(shift.Order.Type.MARKET_SELL,
                                self.symbol, int(item.get_long_shares()/100))
            trader.submit_order(order)
            sleep(2)
            item = trader.get_portfolio_item(self.symbol)
        while item.get_short_shares():
            order = shift.Order(shift.Order.Type.MARKET_SELL,
                                self.symbol, int(item.get_short_shares()/100))
            trader.submit_order(order)
            sleep(2)
            item = trader.get_portfolio_item(self.symbol)
        trader.cancel_all_pending_orders()
        sleep(1)
        trader.disconnect()


if __name__ == "__main__":
    t = Simple_Traders_Simulator([5, 3, 3, 1, 0], 1)
    t.start_sim()