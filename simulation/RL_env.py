from threading import Thread, Lock
import pandas as pd
import numpy as np
import time
import os
import shift
from action_patterns import Action_Patterns
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

"""
class OrderBook:
    def __init__(self, AskOrder, BidOrder, last_price):
        self.last_price = last_price
        idx = 0
        tmp = pd.DataFrame(columns=['price', 'size', 'type', 'last_price'])
        ls = AskOrder

        for order in ls[::-1]:
            tmp.loc[idx] = [order.price, order.size, 'Ask', last_price]
            idx += 1

        self.n_ask = idx
        ls = BidOrder

        for order in ls:
            tmp.loc[idx] = [order.price, order.size, 'Bid', last_price]
            idx += 1

        self.n_bid = idx - self.n_ask

        self.df = tmp

    def __repr__(self):
        return str(self.df)
"""

class SHIFT_env:
    orders = (shift.Order.Type.MARKET_SELL, 
              shift.Order.Type.MARKET_BUY, 
              shift.Order.Type.LIMIT_SELL, 
              shift.Order.Type.LIMIT_BUY)
    
    _default_ZI_param = {'max_order_size': 5, 
                             'order_number': 5,
                             'trade_percent': 0.05,
                             'lambda': 0. }

    def __init__(self,
                 trader,
                 t,
                 nTimeStep,
                 ODBK_range,
                 symbol,
                 target_price,
                 commission = 0,
                 rebate = 0,
                 save_data = True, 
                 max_order_list_length = 50, 
                 ZI_param = {}):
        

        
        #newly added##############
        #objective: control the mid price
        self.target_mid_price = target_price
        self.ZI_param = deepcopy(self._default_ZI_param)
        self.ZI_param.update(ZI_param)
        #bp threshold:
        self.bp_thres = 50000

        self.timeInterval = t
        self.symbol = symbol
        self.nTimeStep = nTimeStep
        self.ODBK_range = ODBK_range
        self.trader = trader
        self.commission = commission
        self.rebate = rebate
        self.mutex = Lock()
        self.strategy_mutex = Lock()
        self.order_mutex = Lock()
        

        self.data_thread = Thread(target=self._link)
        self.strategy_thread = Thread(target = self._strategy_execute)
        # features = ['symbol', 'orderPrice', 'orderSize', 'OrderTime']
        self.table = CirList(nTimeStep) #contains: 'curr_mp', 'volume_ask', 'volume_bid', 'remained_bp'
        self._cols = ['reward', 'order_type', 'price', 'size','curr_mp', 'volume_ask', 
                        'volume_bid', 'remained_bp', 'past_mean_mp', 'mp_vol']
        #['BA_spead', 'last_traded_price', 'Smart_price', 'Liquidity_imb', 'market_cost',
        #        'remained_shares', 'remained_time', 'reward', 'order_type','is_buy', 'premium',
        #        'obj_price', 'base_price', 'executed', 'done']
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
        self.strategy_thread.start()
        

        
        self.action_space, self.action_list = self.create_action_pattern([0.05, 0.1, 0.2, 0.4])
        self.observation_space = gym.spaces.Box(np.array([-np.inf, -np.inf, -np.inf, -np.inf]), 
                                               np.array([np.inf, np.inf, np.inf, np.inf]), dtype = np.float32)
        #new:
        self.current_state_info = self.compute_state_info()
    """
    def set_objective(self, share, remained_time, premium = None):
        self.remained_share = abs(share)
        self.total_share = self.remained_share
        self.currentPos = self._getCurrentPosition()
        self.objPos = self.currentPos + share
        self.remained_time = remained_time
        self.isBuy = True if share> 0 else False
        self.premium = premium if premium else remained_time / 100
    """
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

            #last_price = self.trader.getLastPrice(self.symbol)

            Ask_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.LOCAL_ASK, self.ODBK_range)
            # assert Ask_ls, f'getOrderBook: return empty list: {self.symbol}-ASK-{self.ODBK_range}'

            Bid_ls = self.trader.get_order_book(self.symbol, shift.OrderBookType.LOCAL_BID, self.ODBK_range)
            # assert Bid_ls, f'getOrderBook: return empty list: {self.symbol}-BID-{self.ODBK_range}'
            
            #get remaining buying power
            bp = self.trader.get_portfolio_summary().get_total_bp()

            #get best bid and ask prices
            best_p = self.trader.get_best_price(self.symbol)

            info = self.LOB_to_list(Ask_ls, Bid_ls, best_p, bp)

            self.mutex.acquire()
            # print(88)
            self.table.insertData(info)
            # print(tmp)
            self.mutex.release()

            time.sleep(self.timeInterval)
        print('Data Thread stopped.')
    
    def LOB_to_list(self, ask_book, bid_book, best_p, bp): #return a list with these info 'curr_mp', 'volume_ask', 'volume_bid', 'remained_bp'
        #spread
        #sp = round((best_p.get_ask_price() - best_p.get_bid_price()),3)
        
        #mid price
        mid = round(((best_p.get_ask_price()+best_p.get_bid_price())/2),3)

        bid_size = 0
        ask_size = 0
        for order in ask_book:
            ask_size += order.size
        for order in bid_book:
            bid_size += order.size
        return [mid, ask_size, bid_size, bp]

    def compute_state_info(self):
        #return the following items 'curr_mp', 'volume_ask', 'volume_bid', 'remained_bp', 'past_mean_mp', 'mp_vol'
        tab = self.table
        while True:
            if tab.isFull():
                his_mp = []
                for ele in tab.getData():
                    his_mp.append(ele[0])
                his_mp_np = np.array(his_mp)
                past_mean_mp = np.mean(his_mp_np)
                mp_vol = np.std(his_mp_np)
                return (tab.getData()[self.nTimeStep-1] + [past_mean_mp, mp_vol])
            else:
                print("need to wait for table to fill up")
                time.sleep(1)
        


    def create_action_pattern(self, vols):
        '''
        Return:
        -------
        action_space: gym action space
        action_list: list, a list of strategy functions. 
        '''
        tmp = Action_Patterns()
        action_list = []
        for vol in vols:
            action_list.append(tmp.generate_ZI_trader( \
                                   ticker = self.symbol, 
                                   initial_price = self.target_mid_price,
                                   initial_volatility = vol, 
                                   max_order_size = self.ZI_param['max_order_size'], 
                                   order_number = self.ZI_param['order_number'], 
                                   lambda_ = self.ZI_param["lambda"], 
                                   max_order_list_length = self.max_order_list_length, 
                                   trade_percent=self.ZI_param['trade_percent']))
        action_space = gym.spaces.Discrete(len(vols))
        return action_space, action_list
    
    def load_action_by_index(self, index=None):
        if self.current_action_index == index:
            return 
        self.strategy_mutex.acquire()
        if index is None: 
            self.strategy = self.void_action
        else:
            self.strategy = self.action_list[index]
        self.strategy_mutex.release()
        self.current_action_index = index
        print(f"changed strategy to {index}")
        

    def step(self, action):
        self.load_action_by_index(action)
        
        time.sleep(self.timeInterval)
        
        #STATES: 4 components: ##################################################################################################################
        #    target mid price and real mid price diff | price stability | ask-bid balance | is bp at risk 
        state_info = self.compute_state_info() #: 'curr_mp', 'volume_ask', 'volume_bid', 'remained_bp', 'past_mean_mp', 'mp_vol'
        self.current_state_info = state_info
        state = self.get_states()  #: [mp_diff, mp_vol, ab_bal, is_bp_risk]
        print(state)

        #calculate reward: ###########################################################################################################################################
        reward =     0.75 * max(0, (10 - state[0]))    +     0.25 * max(0, 10 * (1 - abs(0.4 - state[1])))    -   0  * 10 #state[3]
        print(f"Reward: {reward}")

        done = False

        # #save
        # if self.isSave:
        #     #['reward', 'order_type', 'price', 'size','curr_mp', 'volume_ask', 'volume_bid', 'remained_bp', 'past_mean_mp', 'mp_vol']
        #     tmp = [reward, order_type, size , price] + state_info

        #     self.df.loc[self.df_idx] = tmp
        #     self.df_idx += 1

        return state, reward, done, dict()
        
        
        
        
    def _strategy_execute(self):
        print(f"Strategy thread starts")
        self.strategy_thread_alive = True
        while self.trader.is_connected() and  self.strategy_thread_alive:
            self.strategy(self.order_list, self.trader, self.target_mid_price)
            time.sleep(2)
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
                    print(f"remove order: {o.type, o.price, o.size, o.status, o.id}, order list length: {len(self.order_list)}")
                    break
            print(f"{len(self.order_list)}/{self.max_order_list_length}")
            if len(self.order_list) >= self.max_order_list_length:
                for i, order_id in enumerate(self.order_list[:int(self.max_order_list_length/2)]):
                    order = self.trader.get_order(order_id)
                    self.trader.submit_cancellation(order)
                    # print(f"try to cancel: {order_id}")
                    
            time.sleep(0.5)
        print("Order cancel thread done. ")
                    
            
    
    def kill_strategy_thread(self):
        self.strategy_thread_alive = False
        
    def kill_order_thread(self):
        self.order_thread_alive = False
        
        
        
        
        

    def step_(self, order_type, size, price = "na"): #None if no action
        #apply action:##############################################################################################################################################
        #order_type: 1(Mar Sell) 2(Mar Buy) 3(Lmt Sell) 4(Lmt Buy)
        if order_type != None:
            if order_type <= 2:
                order = shift.Order(SHIFT_env.orders[order_type-1], self.symbol, size)
                self.trader.submit_order(order)
            else:
                order = shift.Order(SHIFT_env.orders[order_type-1], self.symbol, size, price)
                self.trader.submit_order(order)
        #wait for the order to have effect on LOB
        time.sleep(self.timeInterval)

        #cancell orders if the orders are not on the best levels: ??????????????

        #STATES: 4 components: ##################################################################################################################
        #    target mid price and real mid price diff | price stability | ask-bid balance | is bp at risk 
        state_info = self.compute_state_info() #: 'curr_mp', 'volume_ask', 'volume_bid', 'remained_bp', 'past_mean_mp', 'mp_vol'
        self.current_state_info = state_info
        state = self.get_states()  #: [mp_diff, mp_vol, ab_bal, is_bp_risk]
        print(state)

        #calculate reward: ###########################################################################################################################################
        reward =     0.75 * max(0, (10 - state[0]))    +     0.25 * max(0, 10 * (1 - abs(0.4 - state[1])))    -   state[3]  * 10 
        print(f"Reward: {reward}")

        terminate = False

        if state[3] <= 100:
            truncate = True
        else: truncate = False
        
        #save
        if self.isSave:
            #['reward', 'order_type', 'price', 'size','curr_mp', 'volume_ask', 'volume_bid', 'remained_bp', 'past_mean_mp', 'mp_vol']
            tmp = [reward, order_type, size , price] + state_info

            self.df.loc[self.df_idx] = tmp
            self.df_idx += 1

        return state, reward, terminate, truncate, dict()

    def get_states(self):
        #target mid price and real mid price diff
        mp_diff = 0.6 * (self.current_state_info[0] - self.target_mid_price) + 0.4 * (self.current_state_info[4] - self.target_mid_price)

        #price stability
        mp_vol = self.current_state_info[5]

        #ask-bid balance:
        ab_bal = self.current_state_info[1] - self.current_state_info[2]

        #is bp at risk
        is_bp_risk = 0
        if self.current_state_info[3] <= self.bp_thres:
            is_bp_risk = 1
        
        return np.array([mp_diff, mp_vol, ab_bal, is_bp_risk])
 
    def reset(self):
        # print(f'Holding shares: {self.trader.getPortfolioItem(self.symbol).getShares()}')
        # print(f'Buying Power: {self.trader.getPortfolioSummary().getTotalBP()}')
        # self.close_all()
        
        self.load_action_by_index()
        self.cancel_all()
        return self.get_states()
    
    def cancel_all(self):
        
        self.load_action_by_index()
        self.trader.cancel_all_pending_orders()
        for _ in range(5):
            for i, order_id in enumerate(self.order_list):
                order = self.trader.get_order(order_id)
                # print(type(order.status)
                self.trader.submit_cancellation(order)
            time.sleep(1)
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

            time.sleep(0.5)
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
