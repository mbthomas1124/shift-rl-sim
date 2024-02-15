from threading import Thread, Lock
import pandas as pd
import numpy as np
import time
import os
import shift
import gymnasium as gym
from copy import deepcopy
from datetime import datetime

from threading import Thread, Lock
import pandas as pd
import numpy as np
from time import sleep
import shift
import gymnasium as gym
from collections import deque

import threading

class lt_env(gym.Env):
    def __init__(
        self,
        trader,
        symbol,
        agent_id,
        state,
        step_time=2,
        order_size=8,
        target_buy_sell_flows = [[0.5], [0.3]],
        switch_steps = 1000,
        risk_aversion=0.5,
        pnl_weighting=0.5,
        normalizer=0.01,
        order_book_range=5,
        max_iterations=None,
    ):
        self.agent_id = agent_id
        self.trader = trader
        self.symbol = symbol
        self.step_time = step_time
        self.order_bp = order_size*100*100
        self.order_size = order_size
        self.target_buy_sell_flows = target_buy_sell_flows
        self.targ_buy_frac = self.target_buy_sell_flows[0][0]
        self.targ_sell_frac = self.target_buy_sell_flows[1][0]
        self.switch_steps = switch_steps
        self.gamma = risk_aversion
        self.alpha = normalizer
        self.w = pnl_weighting
        self.order_book_range = order_book_range
        self.max_iters = max_iterations

        self.state = state
        # mutex
        self.mutex = Lock()

        # data thread
        self.time_step = 0.1
        self.n_time_step = 2 * int(self.step_time / self.time_step)
        #self.data_thread = Thread(target=self._data_thread)
        # contains: mp, sp, inv, bp, w, gamma, bid_volumes, ask_volumes
        self.midprice_list = deque(maxlen=self.n_time_step)
        self.data_thread_alive = True
        #self.data_thread.start()
        self.order_queue = deque(maxlen=100)
        self.buy_count = 0
        self.sell_count = 0
        self.steps_elapsed = 0
        self.prev_q = (self.targ_buy_frac + self.targ_sell_frac) / 2

        # actions
        self.void_action = lambda *args: None
        self.action_space, self.action_list, self.action_params = self.create_actions()

        # states
        # return a list with these values: mp, sp, inv, bp, bid_volumes, ask_volumes, order_size, target order proportions, gamma, alpha, w
        self.observation_space = gym.spaces.Box(
            np.array(
                ([-np.inf] * self.n_time_step)
                + [0, -np.inf]
                + ([0] * self.order_book_range * 2)
                + [0] * 6
            ),
            np.array(
                ([np.inf] * self.n_time_step)
                + [np.inf, np.inf]
                + ([np.inf] * self.order_book_range * 2)
                + [np.inf] * 6
            ),
        )

        # track stats
        self.stats = {}
        self.stats["act_dir"] = []



        # reward trackers
        self.initial_pnl = self.trader.get_portfolio_item(self.symbol).get_realized_pl()# + self.trader.get_unrealized_pl(symbol=self.symbol)
        #self.initial_mp = self.get_state()[self.n_time_step - 1]
        self.initial_inv_pnl = 0

        # while self.trader.get_last_price(self.symbol) == 0:
        #     #sleep(1)
        #     print("LT waiting")

    def lp_action(self, ticker, signal):
        def func(trader):
            order_id = None
            self.order_size = round(self.order_bp / (trader.get_last_price(ticker) * 100))
            if signal == 1:
                # buy
                if trader.get_portfolio_summary().get_total_bp() > (
                    self.order_size * 100 * trader.get_last_price(ticker)
                ):
                    order = shift.Order(
                        shift.Order.Type.MARKET_BUY, ticker, self.order_size
                    )
                    trader.submit_order(order)
                    order_id = order.id
                else:
                    print(
                        f"{ticker} insufficient buying power: {trader.get_portfolio_summary().get_total_bp()}"
                    )

            elif signal == -1:
                # sell
                long_shares = trader.get_portfolio_item(ticker).get_long_shares() > 0
                if long_shares < self.order_size:
                    # involves short selling
                    short_size = self.order_size - long_shares
                    if trader.get_portfolio_summary().get_total_bp() > (
                        short_size * 2 * 100 * trader.get_last_price(ticker)
                    ):
                        order = shift.Order(
                            shift.Order.Type.MARKET_SELL, ticker, self.order_size
                        )
                        trader.submit_order(order)
                        order_id = order.id
                    else:
                        print(
                            f"{ticker} insufficient buying power: {trader.get_portfolio_summary().get_total_bp()}"
                        )
                else:
                    order = shift.Order(
                        shift.Order.Type.MARKET_SELL, ticker, self.order_size
                    )
                    trader.submit_order(order)
                    order_id = order.id

            return order_id

        return func

    def create_actions(self):
        """
        Return:
        -------
        action_space: gym action space
        action_list: list: a list of functions to place orders.
        action_params: a list of tuples of the parameters set by a given action
        """
        action_list = []
        action_params = []
        for act in [0, -1, 1]:
            action_list.append(self.lp_action(ticker=self.symbol, signal=act))
            action_params.append(act)
        action_space = gym.spaces.Discrete(3)
        return action_space, action_list, action_params

    def execute_action(self, action):
        if action == None:
            return None
        else:
            place_order = self.action_list[action]
            order_id = place_order(self.trader)
            return order_id

    def check_order(self, order_id):
        # return size and type of filled order
        # if order was not filled, cancel it and return (0, None)
        order = self.trader.get_order(order_id)
        # print(order.status)
        if order.status == shift.Order.Status.FILLED:
            return True
        else:
            return False

    def _data_thread(self):
        # thread constantly collecting midprice data
        print(f"Data thread starts")

        while self.trader.is_connected() and self.data_thread_alive:
            best_price = self.trader.get_best_price(self.symbol)
            best_bid = best_price.get_bid_price()
            best_ask = best_price.get_ask_price()
            if (best_bid == 0) and (best_ask == 0):
                if len(self.midprice_list) > 0:
                    self.midprice_list.append(self.midprice_list[-1])
            elif (best_bid == 0) or (best_ask == 0):
                self.midprice_list.append(max(best_bid, best_ask))
            else:
                self.midprice_list.append((best_bid + best_ask) / 2)

            #sleep(self.time_step)

        print("Data Thread stopped.")

    def get_state(self):
        # return a list with these values: mp, sp, inv, bp, bid_volumes, ask_volumes, order_size, target order proportions, gamma, alpha, w

        while True:
            if len(self.midprice_list) == self.n_time_step:
                best_price = self.trader.get_best_price(self.symbol)
                best_bid = best_price.get_bid_price()
                best_ask = best_price.get_ask_price()
                if (best_bid == 0) or (best_ask == 0):
                    spread = 0
                else:
                    spread = best_ask - best_bid

                inv = self.trader.get_portfolio_item(self.symbol).get_shares() // 100

                bid_book = self.trader.get_order_book(
                    self.symbol, shift.OrderBookType.LOCAL_BID, self.order_book_range
                )
                ask_book = self.trader.get_order_book(
                    self.symbol, shift.OrderBookType.LOCAL_ASK, self.order_book_range
                )

                bid_levels = []
                ask_levels = []
                for level in bid_book:
                    bid_levels.append(level.size)
                if len(bid_book) < self.order_book_range:
                    bid_levels += [0] * (self.order_book_range - len(bid_book))
                for level in ask_book:
                    ask_levels.append(level.size)
                if len(ask_book) < self.order_book_range:
                    ask_levels += [0] * (self.order_book_range - len(ask_book))

                state = (
                    list(self.midprice_list)
                    + [spread, inv]
                    + bid_levels
                    + ask_levels
                    + [
                        self.order_size,
                        self.targ_buy_frac,
                        self.targ_sell_frac,
                        self.gamma,
                        self.alpha,
                        self.w,
                    ]
                )
                return np.array(state)
            else:
                print("waiting for to collect more data")
                #sleep(1)

    def step(self, action):
        state = [
            99.965, 99.965, 99.97, 99.965, 99.97,
            99.96,99.97, 99.975, 99.955, 99.96,
            99.96,99.97, 99.965, 99.965, 99.96,
            0.99600000e+02,  0.99750000e+02,  0.99350000e+02,  0.99300000e+02, 0.99200000e+02, 
            0.02000000e+00,  2.76271186e+03, 
            3.80000000e+01,  7.00000000e+01,  3.20000000e+01,  0.00000000e+00, 0.00000000e+00,
            1.30000000e+01, 1.14000000e+02,  4.50000000e+01,  4.90000000e+01,  2.29000000e+02] + [
                        self.order_size,
                        self.targ_buy_frac,
                        self.targ_sell_frac,
                        self.gamma,
                        self.alpha,
                        self.w,
                    ]
        state = self.state + [
                        self.order_size,
                        self.targ_buy_frac,
                        self.targ_sell_frac,
                        self.gamma,
                        self.alpha,
                        self.w,
                    ]
        self.stats["act_dir"].append(action)
        return np.array(state), 0, False, False, dict()

    def close_positions(self):
        # close all positions for given ticker
        print("running close positions function for", self.symbol)

        # close any long positions
        item = self.trader.get_portfolio_item(self.symbol)
        long_shares = item.get_long_shares()
        if long_shares > 0:
            print(f"{self.symbol} market selling because long shares = {long_shares}")
            # order = shift.Order(shift.Order.Type.MARKET_SELL, self.symbol, long_shares)
            # self.trader.submit_order(order)
            # #sleep(0.2)
            rejections = 0
            while item.get_long_shares() > 0:
                order = shift.Order(
                    shift.Order.Type.MARKET_SELL, self.symbol, long_shares
                )
                self.trader.submit_order(order)
                #sleep(2)
                if (
                    self.trader.get_order(order.id).status
                    == shift.Order.Status.REJECTED
                ):
                    rejections += 1
                else:
                    break
                if rejections == 5:
                    # if orders get rejected 5 times, just give up
                    break
                # item = self.trader.get_portfolio_item(self.symbol)

        # close any short positions
        item = self.trader.get_portfolio_item(self.symbol)
        short_shares = item.get_short_shares()
        if short_shares > 0:
            print(f"{self.symbol} market buying because short shares = {short_shares}")
            # order = shift.Order(shift.Order.Type.MARKET_BUY, self.symbol, long_shares)
            # self.trader.submit_order(order)
            # #sleep(0.2)
            rejections = 0
            while item.get_short_shares() > 0:
                order = shift.Order(
                    shift.Order.Type.MARKET_BUY, self.symbol, short_shares
                )
                self.trader.submit_order(order)
                #sleep(2)
                if (
                    self.trader.get_order(order.id).status
                    == shift.Order.Status.REJECTED
                ):
                    rejections += 1
                else:
                    break
                if rejections == 5:
                    # if orders get rejected 5 times, just give up
                    print(f"{self.symbol} could not complete close positions")
                    break
                # item = self.trader.get_portfolio_item(self.symbol)

    def reset(self):
        # self.trader.cancel_all_pending_orders()
        # self.close_positions()
        self.buy_count = 0
        self.sell_count = 0
        self.steps_elapsed = 0
        state = [
            99.965, 99.965, 99.97, 99.965, 99.97,
            99.96,99.97, 99.975, 99.955, 99.96,
            99.96,99.97, 99.965, 99.965, 99.96,
            0.99600000e+02,  0.99750000e+02,  0.99350000e+02,  0.99300000e+02, 0.99200000e+02, 
            0.02000000e+00,  2.76271186e+03, 
            3.80000000e+01,  7.00000000e+01,  3.20000000e+01,  0.00000000e+00, 0.00000000e+00,
            1.30000000e+01, 1.14000000e+02,  4.50000000e+01,  4.90000000e+01,  2.29000000e+02] + [
                        self.order_size,
                        self.targ_buy_frac,
                        self.targ_sell_frac,
                        self.gamma,
                        self.alpha,
                        self.w,
                    ]
        state = self.state + [
                        self.order_size,
                        self.targ_buy_frac,
                        self.targ_sell_frac,
                        self.gamma,
                        self.alpha,
                        self.w,
                    ]
        return state, dict()

    def hard_reset(self):
        self.reset()
        for key in self.stats.keys():
            self.stats[key] = []

    def kill_thread(self):
        self.data_thread_alive = False

    def save_to_csv(self, epoch):
        print(f"lt{self.agent_id} mean", np.mean(self.stats["act_dir"]))
        #print("std", np.std(self.stats["act_dir"]))
        df = pd.DataFrame.from_dict(self.stats)
        df.to_csv(epoch, index=False)

    def _getCurrentPosition(self):
        return int(self.trader.getPortfolioItem(self.symbol).getShares())

    def __del__(self):
        self.kill_thread()

    def __call__(self, *args, **kwds):
        return self


class CirList:
    def __init__(self, length):
        self.size = length
        self._table = [None]*length
        self.idx = 0
        self._counter = 0

    def insertData(self, data):
        self._counter += 1
        self._table[self.idx] = data
        self.idx = (self.idx+1) % self.size

    def getData(self):
        tail = self._table[0:self.idx]
        head = self._table[self.idx:]
        ret = head+tail
        return ret.copy()

    def isFull(self):
        return self._counter >= self.size

    def __repr__(self):
        return str(self.getData())


class Env(gym.Env):
    orders = (shift.Order.Type.MARKET_SELL, 
              shift.Order.Type.MARKET_BUY, 
              shift.Order.Type.LIMIT_SELL, 
              shift.Order.Type.LIMIT_BUY)


    def __init__(self,
                 trader,
                 rl_t,
                 info_step,
                 nTimeStep,
                 ODBK_range,
                 symbol,
                 agent_id,
                 state,
                 action_sym = [-1,1],
                 commission = 0,
                 rebate = 0,
                 save_data = True, 
                 max_order_list_length = 20,
                 natural_stop = None,
                 #identifiers
                 alpha = 0.09,
                 weight = 0.5,
                 gamma = 0.09,
                 tar_m = 0.8,
                 
                 #self generate LOB
                 start_price = 100,
                 init = False):
        
        self.agent_id = agent_id
        self.state = state
        #newly added##############
        #objective: control the mid price
        #self.target_mid_price = target_price

        #bp threshold:
        self.bp_thres = 50000
        self.init_bp = trader.get_portfolio_summary().get_total_bp()

        self.rl_t = rl_t
        self.info_step = info_step
        self.symbol = symbol
        self.nTimeStep = nTimeStep
        self.ODBK_range = ODBK_range
        self.trader = trader
        self.commission = commission
        self.rebate = rebate
        self.mutex = Lock()
        # self.strategy_mutex = Lock()
        self.order_mutex = Lock()
        
        #initialize LOB           #keep this TRUE if simulating mm only#####
        self.init = init
        
        self.last_available_price = 100#record last price if price has a huge jump
        self.last_available_sp = 0.01

        # self.data_thread = Thread(target=self._link)
        # features = ['symbol', 'orderPrice', 'orderSize', 'OrderTime']
        self.table = CirList(nTimeStep) #contains: 'curr_mp', 'volume_ask', 'volume_bid', 'remained_bp'
        self._cols = ['action_size', 'action_sym', 'action_asym']

        self.df = pd.DataFrame(columns=self._cols)
        self.df_idx = 0
        self.order_cols = ["timestep", "Type", "Status", "Symbol", "Size", "Executed_size", "Price", "Executed Price", "ID"]
        self.order_df = pd.DataFrame(columns = self.order_cols)
        

        print('Waiting for connection', end='')
        for _ in range(5):
            #time.#sleep(1)
            print('.', end='')
        print()

        self.thread_alive = True
        # self.data_thread.start()

        self.remained_share = None
        self.total_share = None
        self.currentPos = None
        self.objPos = None
        self.isBuy = None
        self.remained_time = None
        self.tmp_obs = [None]*7
        self.name = 'exec_one_asset'
        self.isSave = save_data
        
        self.current_action_index = None
        
        self.order_list = []
        self.max_order_list_length = max_order_list_length
        # self.order_thread = Thread(target = self._maintain_order_list)
        # self.order_thread.start()
        
        self.strategy = self.void_action = lambda *args: None
        

        #Actions: ############
        #self.maxsize = 8                  ##############################important for action size#########################################        
        #now it's % instead of action order size                                #5 to 50 order size
        self.action_space= gym.spaces.Box(low = np.array([0.000005, action_sym[0], -1]), high = np.array([0.00005, action_sym[1], 1]), shape = (3,)) #gym.spaces.Box(low = np.array([1]), high = np.array([5])) #dtype=np.float32, 
        self.observation_space = gym.spaces.Box(np.array([0,0,0,0, 0,0,0,0,0,  -np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                               np.array([0,0,0,0,
                                                         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                                         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                                         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                                         np.inf, np.inf, np.inf]), dtype = np.float64)
        

        #new:
        #self.current_state_info = self.compute_state_info()
        self.total_pl = 0
        self.alpha = alpha    #to normalize pnl part of the reward
        self.w = weight        #weight on pnl
        self.gamma = gamma    #inventory risk aversion
        self.target_market_share = tar_m  #m*

        self.step_counter = 0
        self.natural_stop = natural_stop
        self.last_market_share = 0 
        self.last_inventory_pnl = 0
    
        
    # @staticmethod
    # def action_space():
    #     return 3

    # @staticmethod
    # def state_space():
    #     return 4

    #thread constantly collecting order book data and Last Price
    def _link(self):
        print(f"Data thread starts")
        while self.trader.is_connected() and self.thread_alive:

            item = self.trader.get_portfolio_item(self.symbol)

            Ask_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.LOCAL_ASK, self.ODBK_range)
            # assert Ask_ls, f'getOrderBook: return empty list: {self.symbol}-ASK-{self.ODBK_range}'

            Bid_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.LOCAL_BID, self.ODBK_range)
            # assert Bid_ls, f'getOrderBook: return empty list: {self.symbol}-BID-{self.ODBK_range}'
            
            #get remaining buying power
            bp = self.trader.get_portfolio_summary().get_total_bp()

            #get best bid and ask prices
            best_p = self.trader.get_best_price(self.symbol)

            info = self.LOB_to_list(Ask_ls, Bid_ls, best_p, bp, item)

            self.mutex.acquire()
            # print(88)
            self.table.insertData(info) #table info: [mid, ask_size, bid_size, bp, inventory]
            # print(tmp)
            self.mutex.release()

            #time.#sleep(self.info_step)
        print('Data Thread stopped.')
    
    def LOB_to_list(self, ask_book, bid_book, best_p, bp, item): #return a list with these info 'curr_mp', 'ask_book', 'bid_book', 'remained_bp', inventory, spread, 'ask_book_price', 'bid_book_price' 
        # get LOB
        bid_b = []
        ask_b = []
        bid_p = []
        ask_p = []
        for order in ask_book:
            ask_b.append(order.size) 
            ask_p.append(order.price) 
        for order in bid_book:
            bid_b.append(order.size)
            bid_p.append(order.price)
        
        while len(ask_b) < 5:
            ask_b.append(0)
            ask_p.append(0)
        while len(bid_b) < 5:
            bid_b.append(0)
            bid_p.append(0)
            
        #inventory:
        short_shares = int(item.get_short_shares())
        long_shares = int(item.get_long_shares())
        inventory = long_shares - short_shares
        
        # # #initialize lob:
        if self.init:
            #self.init = False
            return [100, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], self.init_bp, 0, 0.02]
        
        #spread
        sp = abs(round((best_p.get_ask_price() - best_p.get_bid_price()),3))
        
        #mid price
        mid = round(((best_p.get_ask_price()+best_p.get_bid_price())/2),3)
        #check if mid price has a huge jump due to 0 liquidity

        if abs(mid - self.last_available_price) >= 10 or mid == 0 or min(best_p.get_ask_price(),best_p.get_bid_price()) == 0:
            mid = max(self.last_available_price,max(best_p.get_ask_price(),best_p.get_bid_price())) #self.table.getData()[self.nTimeStep-1][0]
            sp = self.last_available_sp#self.table.getData()[self.nTimeStep-1][5]
        self.last_available_price = mid
        self.last_available_sp = sp

        return [mid, ask_b, bid_b, bp, inventory, sp, ask_p, bid_p]

    def compute_state_info(self):
        #return the following items 'curr_mp', 'ask_book', 'bid_book', 'remained_bp', 'current_inventory', spread, "his_prices", 'current_inventory_pnl', total_volume, 'ask_book_price', 'bid_book_price' 
        big_to_small = int(self.rl_t / self.info_step)  #since 5 * 0.4 = 2 which is the time for each step
        tab = self.table
        while True:
            if tab.isFull():
                his_mp = []
                his_inventory = []
                for ele in tab.getData():
                    his_mp.append(ele[0])
                    his_inventory.append(ele[4])
                
                #get the price change at from last timestep
                price_change = his_mp[self.nTimeStep-1]-his_mp[self.nTimeStep - big_to_small]          
                #get the inventory pnl
                current_inventory_pnl = price_change * his_inventory[self.nTimeStep - big_to_small]    #since 5 * 0.4 = 2 which is the time for each step
                
                book = np.asarray(tab.getData()[self.nTimeStep-1][1] + tab.getData()[self.nTimeStep-1][2])
                total_volume = np.sum(book)
                #mean_inven_pnl = np.mean(his_inventory_pnl_np)
                #std_inven_pnl = abs(np.std(his_inventory_pnl_np))
                return (tab.getData()[self.nTimeStep-1][0:6] + [his_mp[5:self.nTimeStep]] + [current_inventory_pnl, total_volume]+tab.getData()[self.nTimeStep-1][6:])
            else:
                print("need to wait for table to fill up")
                #time.#sleep(1)
    
    
    def load_action_by_index(self, actions=None):
        if actions is None:
            return
        else: 
            parameters = actions #self.action_list[index]
            order_size = int(round(self.trader.get_portfolio_summary().get_total_bp() * actions[0] / (100 * self.last_available_price),0))           #order size#self.trader.get_last_price(self.symbol)
            self.generate_mm_trader(self.order_list, self.trader, self.symbol, order_size, parameters[1], parameters[2])
        self.current_action_index = actions
        #print(f"Action: {self.action_list[index]}, {self.symbol}")

    def generate_mm_trader(self, 
                           order_list: list, 
                           trader,
                           ticker: str,
                           order_size: int,    
                           symmetric_e = float,
                            asymmetric_e = float):
        # get necessary data
        
        mid = self.table.getData()[self.nTimeStep-1][0]
        spread = self.table.getData()[self.nTimeStep-1][5]
        #*********************************************************need to ask if ok**********************************************
        spread = self.df["spread"][-20:].mean()
        if spread == 0: spread = 0.01
        #elif spread > 1: spread = 0.01
        # get inventory
        item = trader.get_portfolio_item(ticker)
        imbalance = int(item.get_long_shares()/100 - item.get_short_shares()/100)      #inventory imbalance = long - short
        # print("imbalance", imbalance)
        # print("bp",self.trader.get_portfolio_summary().get_total_bp())
        # submit limit orders
        if order_size > 0:
            p_ask = round((mid + (spread*(0.5*(1+symmetric_e) + asymmetric_e))),2)
            p_bid = round((mid - (spread*(0.5*(1+symmetric_e) - asymmetric_e))),2)
            limit_buy = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, order_size, p_bid)
            limit_sell = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, order_size, p_ask)
            #print("size",order_size)
            order_list.append(limit_buy.id)
            trader.submit_order(limit_buy)
            order_list.append(limit_sell.id)
            trader.submit_order(limit_sell)


    
    def get_states(self):
        #returns:  identifiers | his_prices | current_inventory | last market share | ask book | bid book | different hedge cost | bp
        
        hedge_ratio = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        hf_sp = abs(round((self.current_state_info[5] / 2), 3))
        diff_hedging_cost = ((hf_sp * abs(self.current_state_info[4])) * hedge_ratio).tolist()
        #is bp at risk
        #identifier:
        ident = [self.alpha, self.w, self.gamma, self.target_market_share]
        """
        is_bp_risk = 0
        if self.current_state_info[3] <= self.bp_thres:
            is_bp_risk = 1"""
        #print(np.array(self.current_state_info[6] + [self.current_state_info[4]] + [self.last_market_share] + self.current_state_info[1] + self.current_state_info[2] + diff_hedging_cost))
        return np.array(ident + self.current_state_info[6] + [self.current_state_info[4]] + [self.last_market_share] + self.current_state_info[1] + self.current_state_info[2] + diff_hedging_cost + [100*self.current_state_info[3]/self.init_bp])
    
    def step(self, actions):
        #: identifiers | his_prices | current_inventory | last market share | ask book | bid book | different hedge cost | bp
        state = [9.00000000e-02,  5.00000000e-01,  1.50000000e-01,  5.00000000e-01,
            0.99600000e+02,  0.99750000e+02,  0.99350000e+02,  0.99300000e+02, 0.99200000e+02, 
            6.22000000e+04,  5.76271186e-01, 
            1.30000000e+01, 1.14000000e+02,  4.50000000e+01,  4.90000000e+01,  2.29000000e+02,
            3.80000000e+01,  7.00000000e+01,  3.20000000e+01,  0.00000000e+00, 0.00000000e+00,  
            2.65900009e+02,  5.14750000e+02,  7.63600037e+02,
            9.99301961e+01]
        state = self.state
        self.df.loc[self.df_idx] = [actions[0], actions[1], actions[2]]
        self.df_idx += 1
        return np.array(state), 0, False, False, dict()
            
            
    def _maintain_order_list(self):
        print(f"Order thread starts")
        self.order_thread_alive = True
        while self.trader.is_connected() and  self.order_thread_alive:
            for i, order_id in enumerate(self.order_list):
                order = self.trader.get_order(order_id)
                # print(type(order.status))
                if order.status == shift.Order.Status.CANCELED or \
                   order.status == shift.Order.Status.REJECTED or \
                   order.status == shift.Order.Status.FILLED:
                    popped_order = self.order_list.pop(i)
                    o = self.trader.get_order(popped_order)
                    #print(f"remove order: {o.type, o.price, o.size, o.status, o.id}, order list length: {len(self.order_list)}")
                    break
            #print(f"{len(self.order_list)}/{self.max_order_list_length}")
            if len(self.order_list) >= self.max_order_list_length:
                for i, order_id in enumerate(self.order_list[:int(self.max_order_list_length/2)]):
                    order = self.trader.get_order(order_id)
                    self.trader.submit_cancellation(order)
                    # print(f"try to cancel: {order_id}")
                    
            #time.#sleep(0.1)
        print("Order cancel thread done. ")
                    
        
 
    def reset(self):
        # print(f'Holding shares: {self.trader.getPortfolioItem(self.symbol).getShares()}')
        # print(f'Buying Power: {self.trader.getPortfolioSummary().getTotalBP()}')
        
        # self.load_action_by_index()
        # self.cancel_all()
        # self.close_positions()
        # self.current_state_info = self.compute_state_info() 
        # print(self.get_states())
        state = [9.00000000e-02,  5.00000000e-01,  1.50000000e-01,  5.00000000e-01,
            0.99600000e+02,  0.99750000e+02,  0.99350000e+02,  0.99300000e+02, 0.99200000e+02, 
            6.22000000e+04,  5.76271186e-01, 
            1.30000000e+01, 1.14000000e+02,  4.50000000e+01,  2.29000000e+02,  1.19000000e+03,
            3.80000000e+01,  7.00000000e+01,  3.20000000e+01,  0.00000000e+00, 0.00000000e+00,  
            2.65900009e+02,  5.14750000e+02,  7.63600037e+02,
            9.99301961e+01]
        state = self.state
        return state, dict()
    
    def cancel_all(self):
        
        self.load_action_by_index()
        self.trader.cancel_all_pending_orders()
        for _ in range(5):
            for i, order_id in enumerate(self.order_list):
                order = self.trader.get_order(order_id)
                # print(type(order.status)
                self.trader.submit_cancellation(order)
            #time.#sleep(0.1)
            if len(self.order_list) == 0:
                break
            else:
                print("Tried to cancel existing orders.")
                continue
        print("all order cancelled")

                    

    def save_to_csv(self, filelocation): # TODO replace with logger
        #print("saving")
        #saving the orders submitted
        # step = 0
        # for order in self.trader.get_submitted_orders():
        #     order_info = [order.timestamp, order.type, order.status, order.symbol,order.size, order.executed_size, order.price,
        #         order.executed_price, order.id]
        #     self.order_df.loc[step] = order_info
        #     step += 1
        #     print(step)
        #save the states, reward, LOB infos 
        #try:
        print(f"mm{self.agent_id}", np.mean(self.df["action_size"]), np.mean(self.df["action_sym"]), np.mean(self.df["action_asym"]))
        
        csv_lock = threading.Lock()
        # Create a DataFrame with the data
        data = {
            "agent_ID": [f"mm{self.agent_id} mean"],
            "action_size": [np.mean(self.df["action_size"])],
            "action_sym": [np.mean(self.df["action_sym"])],
            "action_asym": [np.mean(self.df["action_asym"])]
        }
        df = pd.DataFrame(data)

        # Acquire the lock before writing to the CSV file
        with csv_lock:
            try:
                # Try to read the existing CSV file
                existing_df = pd.read_csv("test_action_output.csv")
                # Append the new data
                updated_df = pd.concat([existing_df, df], ignore_index=True)
            except FileNotFoundError:
                # If the file doesn't exist, create a new DataFrame
                updated_df = df

        # Write the updated DataFrame to the CSV file
        updated_df.to_csv("test_action_output.csv", index=False)

        #print("std", np.std(self.df["action_size"]), np.std(self.df["action_sym"]), np.std(self.df["action_asym"]))
        #self.df.to_csv(filelocation)
        #self.df = pd.DataFrame(columns=self._cols)
            #self.order_df.to_csv(f'./iteration_info/Orders_info_{epoch}.csv')
        # except FileNotFoundError:
        #     os.makedirs(f'./iteration_info/', exist_ok= True)
        #     self.df.to_csv(filelocation)
        #     self.df = pd.DataFrame(filelocation)
            #self.order_df.to_csv(f'./iteration_info/Orders_info_{epoch}.csv')


    def kill_thread(self):
        self.thread_alive = False


    def close_positions(self):
        trader = self.trader
        ticker = self.symbol
        # close all positions for given ticker
        print('running close positions function for', ticker)
        
        item = trader.get_portfolio_item(ticker)

        # close any long positions
        long_shares = item.get_long_shares()
        if long_shares > 0:
            print("market selling because long shares =", long_shares)
            order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, int(long_shares/100))
            trader.submit_order(order)
            #time.#sleep(0.2)
            # make sure order submitted correctly
            counter = 0
            item = trader.get_portfolio_item(ticker)
            order = trader.get_order(order.id)
            while (item.get_long_shares() > 0):
                print(order.status)
                #time.#sleep(0.2)
                order = trader.get_order(order.id)
                item = trader.get_portfolio_item(ticker)
                counter += 1
                # if order is not executed after 5 seconds, then break
                if counter > 5:
                    break
                

        # close any short positions
        short_shares = item.get_short_shares()
        if short_shares > 0:
            print("market buying because short shares =", short_shares)
            order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(short_shares/100))
            trader.submit_order(order)
            #time.sleep(0.2)
            # make sure order submitted correctly
            counter = 0
            item = trader.get_portfolio_item(ticker)
            order = trader.get_order(order.id)
            while (item.get_short_shares() > 0):
                print(order.status)
                #time.#sleep(0.2)
                order = trader.get_order(order.id)
                item = trader.get_portfolio_item(ticker)
                counter += 1
                # if order is not executed after 5 seconds, then break
                if counter > 5:
                    break
            
    def __del__(self):
        self.kill_thread()
        
    def __call__(self, *args, **kwds):
        return self
 

"""PPO but trying the parellel subprocess function"""

import argparse
import os
import pprint
from threading import Thread
import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer, AsyncCollector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.discrete import Actor as discrete_Actor, Critic as discrete_Critic
#from shared_policy import shared_env_mm as shared_Env
from time import sleep
import time
import shift
#from lt_rl_env import SHIFT_env as lt_env
from test_flash_crash import flash_crash
import json
import sys

#TODO
#record state action reward next state
#make hidden-size
#research mean field game with shared policy
#mm and lt run
#Sampling the identifiers in a distrbiution so it chagnes while training

def get_args(inputs):
    parser = argparse.ArgumentParser()
    #parser.add_argument('--task', type=str, default='Pendulum-v1')
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=inputs["buffer_size"])
    parser.add_argument('--lr', type=float, default=inputs["lr"])
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epoch', type=int, default=inputs["epoch"])
    parser.add_argument('--step-per-epoch', type=int, default=inputs["step_per_epoch"])
    parser.add_argument('--episode-per-collect', type=int, default=inputs["episode_per_collect"])
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=inputs["batch_size"])
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64]) #make larger
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
    )
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--eps-clip', type=float, default=0.3)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.9)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--resume', action="store_true")

    parser.add_argument("--save-interval", type=int, default=100)
    args = parser.parse_known_args()[0]
    return args


def test_ppo(trader, trader_id, identifiers, agent_type, agent_setup, ticker, state, args, resume_path):
    test_steps = 36000
    start_time = time.time()
    try:
        #########################Market Maker################################
        if agent_type == "mm":
            #initiate envs
            env = Env(trader = trader, symbol=ticker,
                rl_t = agent_setup["rl_t"],
                info_step= agent_setup["info_step"],
                nTimeStep=agent_setup["nTimeStep"],
                ODBK_range=agent_setup["ODBK_range"],
                max_order_list_length=agent_setup["max_order_list_length"],
                weight = identifiers[0],
                gamma = identifiers[1],#increased from 0.09 to 0.15
                tar_m = identifiers[2],#increased from 0.6
                action_sym = identifiers[3],
                agent_id = identifiers[4],
                state = state)
            
            args.state_shape = env.observation_space.shape or env.observation_space.n
            args.action_shape = env.action_space.shape or env.action_space.n
            args.max_action = env.action_space.high[0]
            print(f"MM{identifiers[4] }state_shape{args.state_shape}")
            print(f"MM{identifiers[4] }action_shape{args.action_shape}")
            train_envs = [env]

            net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
            actor = ActorProb(
                net, args.action_shape, max_action=args.max_action, device=args.device
            ).to(args.device)
            critic = Critic(
                Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device),
                device=args.device
            ).to(args.device)
            actor_critic = ActorCritic(actor, critic)
            # orthogonal initialization
            for m in actor_critic.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

            # replace DiagGuassian with Independent(Normal) which is equivalent
            # pass *logits to be consistent with policy.forward
            def dist(*logits):
                #print(*logits)
                #print(Independent(Normal(*logits), 1))
                return Independent(Normal(*logits), 1)
        #########################Liqudity Taker################################
        elif agent_type == "lt":
            # initiate envs
            env = lt_env(trader=trader,  symbol=ticker,
                step_time=agent_setup["step_time"],
                normalizer=agent_setup["normalizer"],
                order_book_range=agent_setup["order_book_range"],

                order_size=identifiers[4],
                target_buy_sell_flows=identifiers[2],
                switch_steps=identifiers[3],
                risk_aversion=identifiers[1], #increased from 0.5
                pnl_weighting=identifiers[0],
                max_iterations=None,
                agent_id = identifiers[5],
                state = state
            )
            train_envs = [env]
            args.state_shape = env.observation_space.shape or env.observation_space.n
            args.action_shape = env.action_space.shape or env.action_space.n
            print(f"state shape: {args.state_shape}")
            print(f"action shape: {args.action_shape}")
            # model
            net = Net(
                args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device,
                softmax=True, norm_layer=torch.nn.InstanceNorm1d, #norm_args=args.state_shape,
            )
            actor = discrete_Actor(net, args.action_shape, device=args.device).to(args.device)
            critic = discrete_Critic(
                net,
                device=args.device
            ).to(args.device)
            actor_critic = ActorCritic(actor, critic)
            # orthogonal initialization
            for m in actor_critic.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

            dist = torch.distributions.Categorical
        

        policy = PPOPolicy(
            actor,
            critic,
            optim,
            dist, #torch.distributions.Normal, 
            discount_factor=args.gamma,
            max_grad_norm=args.max_grad_norm,
            eps_clip=args.eps_clip,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            reward_normalization=args.rew_norm,
            advantage_normalization=args.norm_adv,
            recompute_advantage=args.recompute_adv,
            dual_clip=args.dual_clip,
            value_clip=args.value_clip,
            gae_lambda=args.gae_lambda,
            action_space=env.action_space,
        )
        train_envs = DummyVectorEnv(train_envs)
        # collector
        train_collector = Collector(#Async
            policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs), exploration_noise=True)#
        )
        #test_collector = Collector(policy, test_envs)
        #for loop purpose:
        if len(sys.argv) == 2:
            curr_run_num = sys.argv[1]
            # log
            base_path = "/home/shiftpub/Results_Simulation"
            log_path = os.path.join(base_path, "log")
            if curr_run_num == 1:
                last_checkpoint_path = os.path.join(log_path, f"checkpoint/checkpoint")
            else:
                last_checkpoint_path = os.path.join(log_path, f"checkpoint/checkpoint_{curr_run_num-1}")
            current_checkpoint_path = os.path.join(log_path, f"checkpoint/checkpoint_{curr_run_num}")
            event_path = os.path.join(log_path, "event")
            policy_path = os.path.join(base_path, "Policies")
            iteration_info_path = os.path.join(base_path, f"iteration_info/iteration_info_{curr_run_num}")

            os.makedirs(current_checkpoint_path)
            os.makedirs(iteration_info_path)
                
        else:
            pass
        # log
        base_path = "/home/shiftpub/Results_Simulation"
        log_path = os.path.join(base_path, "log")
        last_checkpoint_path = resume_path
        current_checkpoint_path = os.path.join(log_path, "checkpoint")
        event_path = os.path.join(log_path, "event")
        policy_path = os.path.join(base_path, "Policies")
        iteration_info_path = os.path.join(base_path, "iteration_info")

        #check if the base path exist, if not make those directory
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            os.makedirs(log_path)
            os.makedirs(last_checkpoint_path)
            os.makedirs(event_path)
            os.makedirs(policy_path)
            os.makedirs(iteration_info_path)

        # #remove all files in event folder:
        # files = os.listdir(event_path)
        # for file_name in files:
        #     file_path = os.path.join(event_path, file_name)
        #     if os.path.isfile(file_path):
        #         os.remove(file_path)

        writer = SummaryWriter(event_path)
        logger = TensorboardLogger(writer)#, save_interval=args.save_interval

        def save_best_fn(policy):
            print(f"policy saved_{trader_id}")
            torch.save(policy.state_dict(), os.path.join(policy_path, f"best_policy{trader_id}.pth"))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold

        def save_checkpoint_fn(epoch, env_step, gradient_step):
            ckpt_path = os.path.join(current_checkpoint_path, f"checkpoint_{trader_id}.pth")
            print("saved checkpoint")
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optim": optim.state_dict(),
                }, ckpt_path
            )
            return ckpt_path

        if args.resume:
            # load from existing checkpoint
            print(f"Loading agent under {last_checkpoint_path}")
            ckpt_path = os.path.join(last_checkpoint_path, f"checkpoint_{trader_id}.pth")
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=args.device)
                policy.load_state_dict(checkpoint["model"])
                optim.load_state_dict(checkpoint["optim"])
                print("Successfully restore policy and optim.***************************")
            else:
                print("Fail to restore policy and optim.********************************")
        #trainer
        trainer = OnpolicyTrainer(
            policy,
            train_collector,
            None,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            step_per_collect=5,
            #episode_per_collect=args.episode_per_collect,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            resume_from_log=args.resume,
            save_checkpoint_fn=save_checkpoint_fn,
        )

        for epoch, epoch_stat, info in trainer:
            print(f"Epoch: {epoch}")
            print(epoch_stat)
            print(info)

        #assert stop_fn(info["best_reward"])
        torch.save(policy.state_dict(), os.path.join(policy_path, f"policy{trader_id}.pth"))
        
        if __name__ == "__main__":
            #pprint.pprint(info)
            # Let's watch its performance!
            #env = gym.make(args.task)
            policy.eval()
            #collector = Collector(policy, env)
            print("start colleting result")
            # result = train_collector.collect(n_step = test_steps)
            # print(result)
            #print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
            
    except (KeyboardInterrupt, SystemExit):
        print("thread error catched")
        trader.disconnect()
        env.save_to_csv(f"{iteration_info_path}/sep_trader_{trader_id}.csv")
        torch.save(policy.state_dict(), os.path.join(current_checkpoint_path, f"policy_ppo_temp{trader_id}.pth"))
      
    finally:
        end_time = time.time()
        running_time_seconds = end_time - start_time
        running_time_hours = running_time_seconds // 3600
        running_time_minutes = (running_time_seconds % 3600) // 60
        #print(f"Total running time: {int(running_time_hours)} hours and {int(running_time_minutes)} minutes.")
        env.save_to_csv(f"{iteration_info_path}/sep_trader_{trader_id}.csv")

# def test_ppo_resume(trader, trader_id, w, agent_type, ticker, args):
#     args.resume = True
#     test_ppo(trader, trader_id, w, agent_type, ticker, args)


def test_action(state_mm, state_i, resume_path, trader_list):
    tickers = ["CS1"]#,"CS2"
    config_dic = {
        "agents": ["mm", "mm", "mm", "mm"],#["mm", "mm", "mm", "mm", "lt", "lt", "lt", "lt","lt", "lt", "lt", "lt", "lt","lt"],
        "agent_identifiers": [#MM: weight, gamma,  tar_m,  act_sym_range
                            [0.5,   0.15,   0.5,    [-1,1], 1],                
                            [0.5,   0.15,   0.5,    [-1,1], 2],                
                            [0.5,   0.15,   0.5,    [-1,1], 3],  
                            [0.5,   0.15,   0.5,    [-1,1], 4]],              
                            #[0.5,   0.15,   1,      [-1,2], 4],                
                            # #LT: weight, gamma,  targetBuyFrac, frac_duration, order_size
                            # [0.5,   0.9,    [[0.1],[0.4]], 0,           18, 1],            
                            # [0.5,   0.9,    [[0.1],[0.4]], 0,           18, 2],            
                            # [0.5,   0.9,    [[0.1],[0.4]], 0,            18, 3],            
                            # [0.5,   0.9,    [[0.1],[0.4]], 0,            18, 4],            
                            # [0.5,   0.9,    [[0.1],[0.4]], 0,            18, 5]],            
                            #     [0.5,   0.9,    [[0.2],[0.8]], 0,           18],            
                            #     [0.5,   0.9,    [[0.4],[0.6]], 0,           18],            
                            #     [0.5,   0.9,    [[0.5],[0.5]], 0,            18],            
                            #     [0.5,   0.9,    [[0.6],[0.4]], 0,            18],            
                            #     [0.5,   0.9,    [[0.8],[0.2]], 0,            18]],
        "mm_unified_setup": {"rl_t": 1, "info_step": 0.25, "nTimeStep": 10, "ODBK_range": 5, "max_order_list_length":40},
        "lt_unified_setup": {"step_time": 1, "normalizer":0.01, "order_book_range":5},
        "resume": False,
        "seed": 0,
        "train_args": { "lr": 0.000,
                        "epoch": 1,
                        "step_per_epoch": 1000,
                        "buffer_size": 5,
                        "batch_size": 64,
                        "episode_per_collect": 20},
        "is_flash": False,
        "flash_orders": {"flash_size": 1500,
                         "num_orders": 5,
                         "time_bet_order": 1,
                         "num_flash":88,
                         "time_bet_flash": 400},
    }
    with open('/home/shiftpub/Results_Simulation/iteration_info/config.json', 'w') as json_file:
        json.dump(config_dic, json_file)

    num_proc = len(config_dic["agents"])
    
    threads = []
    #agent identifiers:
    lt_flows = [[0.5, 0.1, 0.5],
                [0.5, 0.1, 0.5]]          
    identifiers = config_dic["agent_identifiers"]
      
    agent_type = config_dic["agents"]
    args = get_args(config_dic["train_args"])
    args.resume = config_dic["resume"]  #resume or not
    torch.manual_seed(config_dic["seed"])
    #: identifiers | his_prices | current_inventory | last market share | ask book | bid book | different hedge cost | bp
    mm_state = [9.00000000e-02,  5.00000000e-01,  1.50000000e-01,  5.00000000e-01,
            0.99600000e+02,  0.99750000e+02,  0.99350000e+02,  0.99300000e+02, 0.99200000e+02, 
            6.22000000e+04,  5.76271186e-01, 
            1.30000000e+01, 1.14000000e+02,  4.50000000e+01,  2.29000000e+02,  1.19000000e+03,
            3.80000000e+01,  7.00000000e+01,  3.20000000e+01,  0.00000000e+00, 0.00000000e+00,  
            2.65900009e+02,  5.14750000e+02,  7.63600037e+02,
            9886851288]
    mm_state = ([9.00000000e-02,  5.00000000e-01,  1.50000000e-01,  5.00000000e-01] + 
                 state_mm["his_prices"][state_i] + state_mm["curr_inv"][state_i] + state_mm["curr_inv"][state_i]
                + state_mm["ask_b"][state_i] + state_mm["bid_b"][state_i] 
                + [2.65900009e+02,  5.14750000e+02,  7.63600037e+02] 
                + state_mm["bp"][state_i])
    # lt_state = [
    #         99.965, 99.965, 99.97, 99.965, 99.97,
    #         99.96,99.97, 99.975, 99.955, 99.96,
    #         99.96,99.97, 99.965, 99.965, 99.96,
    #         0.99600000e+02,  0.99750000e+02,  0.99350000e+02,  0.99300000e+02, 0.99200000e+02, 
    #         0.02000000e+00,  2.76271186e+03, 
    #         3.80000000e+01,  7.00000000e+01,  3.20000000e+01,  0.00000000e+00, 0.00000000e+00,
    #         1.30000000e+01, 1.14000000e+02,  4.50000000e+01,  4.90000000e+01,  2.29000000e+02]
    # mm_state = [9.00000000e-02,  5.00000000e-01,  1.50000000e-01,  5.00000000e-01,
    #         0.99200000e+02,  0.99300000e+02,  0.99350000e+02,  0.99750000e+02, 0.99600000e+02, 
    #         728100,  0.2572811, 
    #         3.80000000e+01,  7.00000000e+01,  3.20000000e+01,  0.00000000e+00, 0.00000000e+00,
    #         1.30000000e+01, 1.14000000e+02,  4.50000000e+01,  4.90000000e+01,  2.29000000e+02,
    #         2.65900009e+02,  5.14750000e+02,  7.63600037e+02,
    #         9886851288]
    # mm_state = [9.00000000e-02,  5.00000000e-01,  1.50000000e-01,  5.00000000e-01,
    #         0.99200000e+02,  0.99200000e+02,  0.99200000e+02,  0.99200000e+02, 0.99200000e+02, 
    #         10000,  0.3072811, 
    #         3.80000000e+01,  7.00000000e+01,  3.20000000e+01,  0.00000000e+00, 0.00000000e+00,
    #         1.30000000e+01, 1.14000000e+02,  4.50000000e+01,  4.90000000e+01,  2.29000000e+02,
    #         2.65900009e+02,  5.14750000e+02,  7.63600037e+02,
    #         9886851288]
    # lt_state = [
    #         99.965, 99.965, 99.97, 99.965, 99.97,
    #         99.96,99.97, 99.975, 99.955, 99.96,
    #         99.96,99.97, 99.965, 99.965, 99.96,
    #         0.99600000e+02,  0.99750000e+02,  0.99350000e+02,  0.99300000e+02, 0.99200000e+02, 
    #         0.01000000e+00,  -4000, 
    #         3.80000000e+01,  7.00000000e+01,  3.20000000e+01,  0.00000000e+00, 0.00000000e+00,
    #         1.30000000e+01, 1.14000000e+02,  4.50000000e+01,  4.90000000e+01,  2.29000000e+02]
    
    for ticker in tickers:
        if ticker == "CS1": mm_index, lt_index, mm_id, lt_id = 0, 0, 0, 0
        elif ticker == "CS2": mm_index, lt_index, mm_id, lt_id = 4, 10, 0, 0
        for i in range(num_proc):
            if agent_type[i] == "mm":
                num = "{:02d}".format(mm_index+1)
                #trader_list.append(shift.Trader(f"marketmaker_rl_{num}"))
                threads.append(Thread(target=test_ppo,args =(trader_list[i], f"mm{mm_id+1}", identifiers[i], agent_type[i], config_dic["mm_unified_setup"], ticker, mm_state, args, resume_path)))        
                print(f"mm{mm_id+1}")
                mm_index += 1
                mm_id += 1
                
            if agent_type[i] == "lt":
                num = "{:02d}".format(lt_index+1)
                trader_list.append(shift.Trader(f"liquiditytaker_rl_{num}"))
                threads.append(Thread(target=test_ppo,args =(trader_list[i], f"lt{lt_id+1}", identifiers[i], agent_type[i], config_dic["lt_unified_setup"], ticker, lt_state, args)))        
                print(f"lt{lt_id+1}")
                lt_index += 1
                lt_id += 1


    
    
    if config_dic["is_flash"]:
        #connect flash crash agent
        crash_maker = shift.Trader("flash_crash_maker_01")
        crash_maker.disconnect()
        crash_maker.connect("/home/shiftpub/initiator.cfg", "password")
        crash_maker.sub_all_order_book()

        flash_config = config_dic["flash_orders"]
        flash_crash_thread = Thread(target=flash_crash, args=(crash_maker, "sell", flash_config["flash_size"], flash_config["num_orders"], flash_config["time_bet_order"],
                                                            flash_config["num_flash"], flash_config["time_bet_flash"], "CS1"))

    # subscribe order book for all tickers
    # start_time = datetime.now().replace(minute = 15, second = 0)
    # while datetime.now() < start_time:
    #     #sleep(1)
    print("Starting")

    for thread in threads:  
        thread.start()
        #sleep(1)
    if config_dic["is_flash"]:
        flash_crash_thread.start()

    for thread in threads:
        thread.join()
    if config_dic["is_flash"]:
        flash_crash_thread.join()
        
    # except KeyboardInterrupt:
    #     trader_list[0].disconnect()

    # finally:
    #     trader_list[0].disconnect()

def collect_data(path):
    #help to collect agents data given the iteration_info path
    csv_files = [file for file in os.listdir(path)]
    sim = {"lt":[], "mm":[]}
    for file in csv_files:
        file_name, e = os.path.splitext(file)
        
        if file_name.startswith("sep_trader_lt"):
            sim["lt"].append(pd.read_csv(os.path.join(path, file)))
        elif file_name.startswith("sep_trader_mm"):
            sim["mm"].append(pd.read_csv(os.path.join(path, file)))
    return sim

if __name__ == "__main__":
    #collect data
    base_path = "/home/shiftpub/sims_final/40_order_len"
    #base2_path = "/home/shiftpub/sims_final/flash"

    flash_train_data=[]
    #for i in range(10):
    path = os.path.join(base_path, f"{0}")
    flash_train_path = os.path.join(path, f"flash_train/iteration_info")
    flash_train_data.append(collect_data(flash_train_path))

    up_states = {"his_prices":[], "curr_inv":[], "last_ms":[],
                "ask_b":[], "bid_b":[], "bp":[]}
    data = flash_train_data[0]["mm"][0]
    count = 0
    # for i in range(4,len(data)):
    #     his_prices = []
    #     for x in range(5):
    #         his_prices.append(data.iloc[i-x]["curr_mp"])
    #     ask_b = [data.iloc[i]["ask_1"], data.iloc[i]["ask_2"], data.iloc[i]["ask_3"], data.iloc[i]["ask_4"], data.iloc[i]["ask_5"]]
    #     bid_b = [data.iloc[i]["bid_1"], data.iloc[i]["bid_2"], data.iloc[i]["bid_3"], data.iloc[i]["bid_4"], data.iloc[i]["bid_5"]]
        
    #     if (np.max(his_prices) - data.iloc[i]["curr_mp"] >= 0.17 and
    #         data.iloc[i]["Ask_depth"] - data.iloc[i]["Bid_depth"] >= 3 and
    #         np.sum(ask_b) - np.sum(bid_b) >= 500):
    #         count+=1
    #         up_states["his_prices"].append(his_prices)
    #         up_states["curr_inv"].append([data.iloc[i]["current_inventory"]])
    #         up_states["last_ms"].append([data.iloc[i-1]["current_market_share"]])
    #         up_states["ask_b"].append(ask_b)
    #         up_states["bid_b"].append(bid_b)
    #         up_states["bp"].append([data.iloc[i]["bp"]])
    for i in range(4,len(data)):
        his_prices = []
        for x in range(5):
            his_prices.append(data.iloc[i-x]["curr_mp"])
        ask_b = [data.iloc[i]["ask_1"], data.iloc[i]["ask_2"], data.iloc[i]["ask_3"], data.iloc[i]["ask_4"], data.iloc[i]["ask_5"]]
        bid_b = [data.iloc[i]["bid_1"], data.iloc[i]["bid_2"], data.iloc[i]["bid_3"], data.iloc[i]["bid_4"], data.iloc[i]["bid_5"]]
        
        if (abs(np.max(his_prices) - data.iloc[i]["curr_mp"]) <= 0.03 and
            data.iloc[i]["Ask_depth"] - data.iloc[i]["Bid_depth"] == 0 and
            abs(np.sum(ask_b) - np.sum(bid_b)) <= 20):
            count+=1
            up_states["his_prices"].append(his_prices)
            up_states["curr_inv"].append([data.iloc[i]["current_inventory"]])
            up_states["last_ms"].append([data.iloc[i-1]["current_market_share"]])
            up_states["ask_b"].append(ask_b)
            up_states["bid_b"].append(bid_b)
            up_states["bp"].append([data.iloc[i]["bp"]])

    try:
        trader_list = []
        for i in range(4):
            num = "{:02d}".format(i+1)
            trader_list.append(shift.Trader(f"marketmaker_rl_{num}"))
        for i in range(len(trader_list)):#len(tickers)*
            trader_list[i].disconnect()
            trader_list[i].connect("/home/shiftpub/initiator.cfg", "password")
            trader_list[i].sub_all_order_book()
            sleep(1)
            print(f"bp of {i+1}:",trader_list[i].get_portfolio_summary().get_total_bp())

        print(count)
        num_states = 5
        num_policies = 2
        for i in range(num_policies * num_states):
            index = int(i/num_states)
            path = f"/home/shiftpub/sims_final/40_order_len/{index}/flash_train/checkpoint"
            test_action(up_states, i%num_states, path, trader_list)

    except KeyboardInterrupt:
        trader_list[0].disconnect()

    finally:
        trader_list[0].disconnect()