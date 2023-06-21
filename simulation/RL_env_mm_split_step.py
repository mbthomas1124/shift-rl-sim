"""
RL mm for shared policy: different from F_RL: step() is split into two parts
"""
from threading import Thread, Lock
import pandas as pd
import numpy as np
import time
import os
import shift
import gym
from copy import deepcopy


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


class SHIFT_env:
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
                 start_price = 100):
        

        
        #newly added##############
        #objective: control the mid price
        #self.target_mid_price = target_price

        #bp threshold:
        self.bp_thres = 50000

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
        
        #initialize LOB
        self.init = True
        
        self.data_thread = Thread(target=self._link)
        # features = ['symbol', 'orderPrice', 'orderSize', 'OrderTime']
        self.table = CirList(nTimeStep) #contains: 'curr_mp', 'volume_ask', 'volume_bid', 'remained_bp'
        self._cols = ['reward', 'action_size', 'action_sym', 'action_asym', 'action_hedge', 'recent_pl_change', 'change_inven_pnl', 
                      'diff_market_share', 'curr_mp', 'current_inventory', 'current_market_share', 'total_volume', 'bp', "spread",
                      "bid_5","bid_4", "bid_3","bid_2","bid_1", "ask_1", "ask_2","ask_3","ask_4","ask_5"]

        self.df = pd.DataFrame(columns=self._cols)
        self.df_idx = 0

        print('Waiting for connection', end='')
        for _ in range(5):
            time.sleep(1)
            print('.', end='')
        print()

        self.thread_alive = True
        self.data_thread.start()

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
        self.order_thread = Thread(target = self._maintain_order_list)
        self.order_thread.start()
        
        self.strategy = self.void_action = lambda *args: None
        

        
        self.action_space= gym.spaces.Box(low = np.array([0.01,-1, -1, 0]), high = np.array([1, 1, 1, 1]), shape = (4,)) #gym.spaces.Box(low = np.array([1]), high = np.array([5])) #dtype=np.float32, 
        #self.create_action_pattern([1, 5, 10, 15, 20], [-1, -0.5, 0, 0.5, 1], [-1, -0.5, 0, 0.5, 1], [0.2, 0.5, 0.8])  , self.action_list 
        self.observation_space = gym.spaces.Box(np.array([0,0,0,0, 0,0,0,0,0,  -np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                                               np.array([0,0,0,0,
                                                         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                                         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                                         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                                         np.inf, np.inf]), dtype = np.float64)
        #new:
        self.current_state_info = self.compute_state_info()
        self.total_pl = 0
        self.alpha = alpha    #to normalize pnl part of the reward
        self.w = weight        #weight on pnl
        self.gamma = gamma    #inventory risk aversion
        self.target_market_share = tar_m  #m*

        self.step_counter = 0
        self.natural_stop = natural_stop
        self.last_market_share = 0 
        self.last_inventory_pnl = 0
        
        
        #generate initial LOB
        #[0, ask_b, bid_b, bp, inventory, sp]
        
        
        
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

            time.sleep(self.info_step)
        print('Data Thread stopped.')
    
    def LOB_to_list(self, ask_book, bid_book, best_p, bp, item): #return a list with these info 'curr_mp', 'ask_book', 'bid_book', 'remained_bp', inventory, spread    
        # get LOB
        bid_b = []
        ask_b = []
        for order in ask_book:
            ask_b.append(order.size) 
        for order in bid_book:
            bid_b.append(order.size)
        
        while len(ask_b) < 5:
                ask_b.append(0)
        while len(bid_b) < 5:
            bid_b.append(0)
            
        #inventory:
        short_shares = int(item.get_short_shares())
        long_shares = int(item.get_long_shares())
        inventory = long_shares - short_shares
        
        # # #initialize lob:
        if self.init:
            #self.init = False
            return [100, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], 50000000.0, 0, 0.02]
        
        #spread
        sp = abs(round((best_p.get_ask_price() - best_p.get_bid_price()),3))
        
        #mid price
        mid = round(((best_p.get_ask_price()+best_p.get_bid_price())/2),3)
        
        #check if mid price has a huge jump due to 0 liquidity
        if sp >= 1:
            mid = self.table.getData()[self.nTimeStep-1][0]
            #print("mid",mid)
            sp = self.table.getData()[self.nTimeStep-1][5]
            #print(sp)

        return [mid, ask_b, bid_b, bp, inventory, sp]

    def compute_state_info(self):
        #return the following items 'curr_mp', 'ask_book', 'bid_book', 'remained_bp', 'current_inventory', spread, "his_prices", 'current_inventory_pnl', total_volume
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
                return (tab.getData()[self.nTimeStep-1][0:6] + [his_mp[5:self.nTimeStep]] + [current_inventory_pnl, total_volume])
            else:
                print("need to wait for table to fill up")
                time.sleep(1)
        


    def create_action_pattern(self, size_ratios, sym_p_e, asym_p_e, hedge):
        '''
        Return:
        -------
        action_space: gym action space
        action_list: list, a list of strategy functions. 
        '''
        #tmp = Action_Patterns()
        total_action_combinations = len(size_ratios) * len(sym_p_e) * len(asym_p_e) * len(hedge)
        action_list = []
        for ratio in size_ratios:
            for s_e in sym_p_e:
                for as_e in asym_p_e:
                    for h in hedge:
                        action_list.append([ratio,s_e,as_e,h])
        
        action_space = gym.spaces.Box(low = np.array([0.01,-1,-1,0]), high = np.array([1,1,1,1]), dtype=np.float32, shape = (4,)) #Discrete(total_action_combinations)
        return action_space, action_list
    
    def load_action_by_index(self, actions=None):
        if actions is None:
            return
        else: 
            parameters = actions #self.action_list[index]
            self.generate_mm_trader(self.order_list, self.trader, self.symbol, int(parameters[0]*100), parameters[1], parameters[2], parameters[3])
        self.current_action_index = actions
        #print(f"Action: {self.action_list[index]}, {self.symbol}")

    def generate_mm_trader(self, 
                           order_list: list, 
                           trader,
                           ticker: str,
                           order_size: int,    
                           symmetric_e = float,
                            asymmetric_e = float,
                            hedge_ratio = float):
        # get necessary data
        
        mid = self.table.getData()[self.nTimeStep-1][0]
        spread = self.table.getData()[self.nTimeStep-1][5]
        #*********************************************************need to ask if ok**********************************************
        if spread == 0: spread = 0.01
        #elif spread > 1: spread = 0.01
                
        # get inventory
        item = trader.get_portfolio_item(ticker)
        imbalance = int(item.get_long_shares()/100) - int(item.get_short_shares()/100)      #inventory imbalance = long - short
        hedge_amount = round(imbalance*hedge_ratio)                

        # submit limit orders
        if order_size > 0:
            p_ask = round((mid + (spread*(0.5*(1+symmetric_e) + asymmetric_e))),2)
            p_bid = round((mid - (spread*(0.5*(1+symmetric_e) - asymmetric_e))),2)

            limit_buy = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, order_size, p_bid)
            limit_sell = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, order_size, p_ask)
            
            order_list.append(limit_buy.id)
            trader.submit_order(limit_buy)
            order_list.append(limit_sell.id)
            trader.submit_order(limit_sell)
            #print("bid/ask size:", order_size_ratio)
            
        # submit market hedge orders
        if hedge_amount > 0:
            market_sell = shift.Order(shift.Order.Type.MARKET_SELL, ticker, hedge_amount)
            order_list.append(market_sell.id)
            trader.submit_order(market_sell)
        elif hedge_amount < 0:
            market_buy = shift.Order(shift.Order.Type.MARKET_BUY, ticker, hedge_amount)
            order_list.append(market_buy.id)
            trader.submit_order(market_buy)

    
    def get_states(self):
        #returns:  identifiers | his_prices | current_inventory | last market share | ask book | bid book | different hedge cost 
        
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
        return np.array(ident + self.current_state_info[6] + [self.current_state_info[4]] + [self.last_market_share] + self.current_state_info[1] + self.current_state_info[2] + diff_hedging_cost)
    
    def step1(self, actions):
        if self.init:
            self.init = False
            limit_buy = shift.Order(shift.Order.Type.LIMIT_BUY, self.symbol, 50, 100-0.02)
            limit_sell = shift.Order(shift.Order.Type.LIMIT_SELL, self.symbol, 50, 100+0.02)    
            self.order_list.append(limit_buy.id)
            self.trader.submit_order(limit_buy)
            self.order_list.append(limit_sell.id)
            self.trader.submit_order(limit_sell)
        
        #print(f"ticker{self.symbol}, action {action}")
        #get current total pl by size:
        self.total_pl = self.trader.get_portfolio_summary().get_total_realized_pl()
        #action = [5,1,0,1]
        self.load_action_by_index(actions)
        #print("actions", actions)

        #print(actions)
        actions_str = str(actions[0]) + ", " + str(actions[1]) + ", " + str(actions[2]) + ", " + str(actions[3])

    def step2(self, actions):        
        #STATES: 7 components: ##################################################################################################################
        #   identifiers | his_prices | current_inventory | last market share | ask book | bid book | different hedge cost 
        state_info = self.compute_state_info() #:'curr_mp', 'ask_book', 'bid_book', 'remained_bp', 'current_inventory', spread, "his_prices", 'current_inventory_pnl', total_volume
        #print(state_info)
        self.current_state_info = state_info
        state = self.get_states()  #: identifiers | his_prices | current_inventory | last market share | ask book | bid book | different hedge cost 
        #print(state)
        #calculate reward: ###########################################################################################################################################
        recent_pl_change = self.trader.get_portfolio_summary().get_total_realized_pl() - self.total_pl
        current_order_sizes = 0
        for order in self.trader.get_waiting_list():
            current_order_sizes += order.size - order.executed_size
        current_market_share = current_order_sizes / state_info[8] #changed to current agent' shares / market total shares, # actions[0]

        diff_market_share = abs((current_market_share - self.target_market_share))# paper may has a typo：  - (self.last_market_share - self.target_market_share)
        change_inven_pnl = abs(state_info[7] - self.last_inventory_pnl)
        reward =  self.w * self.alpha * (recent_pl_change - (self.gamma * change_inven_pnl)) - (1-self.w) * diff_market_share
        #print(self.alpha * (recent_pl_change - (self.gamma * change_inven_pnl)), diff_market_share)
        #print(f"Reward: {reward}")

        curr_mp = state_info[0]
        current_inventory = state_info[4]
        total_volume = state_info[8]
        bp = self.trader.get_portfolio_summary().get_total_bp()
        
        self.last_market_share = current_market_share
        self.last_inventory_pnl = state_info[7]
        
        done = False

        #save
        if self.isSave:
             #['reward', 'action_size', 'action_sym', 'action_asym', 'action_hedge', 'recent_pl_change', 'change_inven_pnl', 'diff_market_share', 
             # 'curr_mp', 'current_inventory', 'current_market_share', 'total_volume', 'bp', "spread","bid_5","bid_4", "bid_3","bid_2","bid_1", "ask_1", "ask_2","ask_3","ask_4","ask_5"]
            tmp = [reward, actions[0], actions[1], actions[2], actions[3], recent_pl_change, change_inven_pnl, 
                   diff_market_share] + [curr_mp, current_inventory, current_market_share, total_volume, 
                                         bp, state_info[5]] + state_info[2][::-1] + state_info[1]
            self.df.loc[self.df_idx] = tmp
            #print(tmp)
            self.df_idx += 1

        if self.natural_stop != None:
            self.step_counter+=1
            if self.step_counter > self.natural_stop:
                done = True
        return state, reward, done, dict()
        
        
        
        
    def _strategy_execute(self):
        print(f"Strategy thread starts")
        self.strategy_thread_alive = True
        while self.trader.is_connected() and  self.strategy_thread_alive:
            self.strategy(self.order_list, self.trader)
            time.sleep(self.strat_t)
        print("Strategy thread done.")
            
            
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
                    
            time.sleep(0.1)
        print("Order cancel thread done. ")
                    
            
    
    def kill_strategy_thread(self):
        self.strategy_thread_alive = False
        
    def kill_order_thread(self):
        self.order_thread_alive = False
        
        

    
 
    def reset(self):
        # print(f'Holding shares: {self.trader.getPortfolioItem(self.symbol).getShares()}')
        # print(f'Buying Power: {self.trader.getPortfolioSummary().getTotalBP()}')
        # self.close_all()
        
        self.load_action_by_index()
        self.cancel_all()
        self.close_positions()
        self.current_state_info = self.compute_state_info() 
        return self.get_states()
    
    def cancel_all(self):
        
        self.load_action_by_index()
        self.trader.cancel_all_pending_orders()
        for _ in range(5):
            for i, order_id in enumerate(self.order_list):
                order = self.trader.get_order(order_id)
                # print(type(order.status)
                self.trader.submit_cancellation(order)
            time.sleep(0.1)
            if len(self.order_list) == 0:
                break
            else:
                print("Tried to cancel existing orders.")
                continue
        print("all order cancelled")

                    

    def save_to_csv(self, epoch): # TODO replace with logger
        try:
            self.df.to_csv(f'./iteration_info/itr_{epoch}.csv')
            self.df = pd.DataFrame(columns=self._cols)
        except FileNotFoundError:
            os.makedirs(f'./iteration_info/', exist_ok= True)
            self.df.to_csv(f'./iteration_info/itr_{epoch}.csv')
            self.df = pd.DataFrame(columns=self._cols)


    def kill_thread(self):
        self.thread_alive = False

 
    """    
    @staticmethod
    def _ba_spread(df, n_ask):
        spread = df.price[n_ask - 1] - df.price[n_ask]
        return spread

    @staticmethod
    def _price(df):
        return df.last_price[0]/1000

    @staticmethod
    def _smart_price(df, n_ask):
        price = (df['size'][n_ask] * df.price[n_ask - 1] + df['size'][n_ask - 1] * df.price[n_ask]) \
                / (df['size'][n_ask] + df['size'][n_ask - 1])
        return price/1000

    @staticmethod
    def _liquid_imbal(df, n_ask, n_bid, act_direction):
        if n_ask > n_bid:
            imbal = df['size'][n_ask:].sum() - df['size'][(n_ask - n_bid):n_ask].sum()
        else:
            imbal = df['size'][n_ask:(2 * n_ask)].sum() - df['size'][0:n_ask].sum()
        if act_direction == 'Sell':
            imbal = -imbal
        return imbal/1000

    @staticmethod
    def _market_cost(df, n_ask, n_bid, act_direction, shares, commission):
        if act_direction == 'Buy':
            counter = df['size'][n_ask-1]
            n_cross = 1
            while counter < shares and n_ask-1 >= n_cross:
                counter += df['size'][n_ask-1-n_cross]
                n_cross += 1
            if n_cross > 1:
                sub_size = np.array(df['size'][(n_ask-n_cross):n_ask])
                sub_price = np.array(df.price[(n_ask-n_cross):n_ask])
                sub_size[0] = shares - sum(sub_size) + sub_size[0]
                market_price = sub_size.dot(sub_price)/shares
                cost = shares*(market_price - df.price[n_ask] + df.price[n_ask-1]*commission)
            else:
                market_price = df.price[n_ask-1]
                cost = shares*(market_price*(1+commission)-df.price[n_ask])
        else:
            counter = df['size'][n_ask]
            n_cross = 1
            while counter < shares and n_cross <= n_bid-1:
                counter += df['size'][n_ask+n_cross]
                n_cross += 1
            if n_cross > 1:
                sub_size = np.array(df['size'][n_ask:(n_ask+n_cross)])
                sub_price = np.array(df.price[n_ask:(n_ask+n_cross)])
                sub_size[-1] = shares - sum(sub_size) + sub_size[-1]
                market_price = sub_size.dot(sub_price)/shares
                cost = shares*(market_price - df.price[n_ask-1] + df.price[n_ask]*commission)
            else:
                market_price = df.price[n_ask]
                cost = shares*(market_price*(1+commission) - df.price[n_ask-1])
        return cost/1000, market_price
    """
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
            time.sleep(0.2)
            # make sure order submitted correctly
            counter = 0
            item = trader.get_portfolio_item(ticker)
            order = trader.get_order(order.id)
            while (item.get_long_shares() > 0):
                print(order.status)
                time.sleep(0.2)
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
            time.sleep(0.2)
            # make sure order submitted correctly
            counter = 0
            item = trader.get_portfolio_item(ticker)
            order = trader.get_order(order.id)
            while (item.get_short_shares() > 0):
                print(order.status)
                time.sleep(0.2)
                order = trader.get_order(order.id)
                item = trader.get_portfolio_item(ticker)
                counter += 1
                # if order is not executed after 5 seconds, then break
                if counter > 5:
                    break
            
    def close_all(self):
        share = self.trader.get_portfolio_item(self.symbol).get_shares()
        BP = self.trader.get_portfolio_summary().get_total_bp()
        waitingStep = 0
        small_order = 1
        while share != 0:
            position = int(share / 100)
            orderType = shift.Order.MARKET_BUY if position < 0 else shift.Order.MARKET_SELL

            if share < 0 and BP < abs(share) * self.trader.get_close_price(self.symbol, True, abs(position)):
                order = shift.Order(orderType, self.symbol, small_order)
                self.trader.submit_order(order)
                small_order *= 2
            else:
                order = shift.Order(orderType, self.symbol, abs(position))
                self.trader.submit_order(order)

            time.sleep(0.1)
            #print(trader.getPortfolioItem(symbol).getShares())
            #print(trader.getPortfolioSummary().getTotalBP())
            share = self.trader.get_portfolio_item(self.symbol).get_shares()
            waitingStep += 1
            assert  waitingStep < 40

    """
    def _getCurrentPosition(self):
        return int(self.trader.getPortfolioItem(self.symbol).getShares() / 100)
    """
    def __del__(self):
        self.kill_thread()
        
    def __call__(self, *args, **kwds):
        return self
        
        
"""

if __name__=='__main__':
    table = CirList(3)
    #self._cols = ['reward', 'order_type', 'price']
    table.insertData([1,2,3])
    table.insertData([1,4,3])
    table.insertData([1,6,3])
    table.insertData([1,6,1])
    print(table.getData())
    print(table._table)
    print(table._counter)
    print(table.isFull())

"""
