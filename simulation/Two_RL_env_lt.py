from threading import Thread, Lock
import pandas as pd
import numpy as np
from time import sleep
import shift
import gymnasium as gym
from collections import deque


class SHIFT_env(gym.Env):
    def __init__(
        self,
        trader,
        symbol = ["CS1", "CS2"],
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

        # mutex
        self.mutex = Lock()

        # data thread
        self.time_step = 0.1
        self.n_time_step = 2 * int(self.step_time / self.time_step)

        #first symbol
        self.data_thread = Thread(target=self._data_thread, args=(self.symbol[0],))
        # contains: mp, sp, inv, bp, w, gamma, bid_volumes, ask_volumes
        self.midprice_list = deque(maxlen=self.n_time_step)
        self.data_thread_alive = True
        self.data_thread.start()
        self.order_queue = deque(maxlen=100)
        self.buy_count = 0
        self.sell_count = 0
        self.steps_elapsed = 0
        self.prev_q = (self.targ_buy_frac + self.targ_sell_frac) / 2

        #second symbol:
        self.data_thread_2 = Thread(target=self._data_thread, args=(self.symbol[1],))
        # contains: mp, sp, inv, bp, w, gamma, bid_volumes, ask_volumes
        self.midprice_list_2 = deque(maxlen=self.n_time_step)
        self.data_thread_2.start()
        self.order_queue_2 = deque(maxlen=100)
        self.buy_count_2 = 0
        self.sell_count_2 = 0
        self.prev_q_2 = (self.targ_buy_frac + self.targ_sell_frac) / 2

        # actions
        self.void_action = lambda *args: None
        self.action_space, self.action_list, self.action_params = self.create_actions()

        # states
        # return a list with these values: mp, sp, inv, bid_volumes, ask_volumes, order_size, target order proportions, gamma, alpha, w
        #                                   mp_2, sp_2, inv_2, bid_volumes_2, ask_volumes_2, order_size
        self.observation_space = gym.spaces.Box(
            np.array(
                ([-np.inf] * self.n_time_step)
                + [0, -np.inf]
                + ([0] * self.order_book_range * 2)
                + [0] * 6

                + ([-np.inf] * self.n_time_step)
                + [0, -np.inf]
                + ([0] * self.order_book_range * 2)
                + [0]
            ),
            np.array(
                ([np.inf] * self.n_time_step)
                + [np.inf, np.inf]
                + ([np.inf] * self.order_book_range * 2)
                + [np.inf] * 6

                + ([np.inf] * self.n_time_step)
                + [np.inf, np.inf]
                + ([np.inf] * self.order_book_range * 2)
                + [np.inf]
            ),
        )

        # track stats
        self.stats = {}
        self.stats["act_dir"] = []
        self.stats["rew"] = []
        self.stats["total_pnl"] = []
        self.stats["pnl"] = []
        self.stats["inv_pnl"] = []
        self.stats["buy_frac"] = []
        self.stats["sell_frac"] = []
        self.stats["curr_q"] = []
        self.stats["mp"] = []
        self.stats["sp"] = []
        self.stats["inv"] = []
        self.stats["bp"] = []
        self.stats["bid_vol"] = []
        self.stats["ask_vol"] = []
        self.stats["filled"] = []

        self.stats["act_dir_2"] = []
        self.stats["inv_pnl_2"] = []
        self.stats["buy_frac_2"] = []
        self.stats["sell_frac_2"] = []
        self.stats["curr_q_2"] = []
        self.stats["mp_2"] = []
        self.stats["sp_2"] = []
        self.stats["inv_2"] = []
        self.stats["bid_vol_2"] = []
        self.stats["ask_vol_2"] = []
        self.stats["filled_2"] = []

        sleep(5)

        # reward trackers
        self.initial_pnl = self.trader.get_portfolio_summary().get_total_realized_pl()# + self.trader.get_unrealized_pl(symbol=self.symbol)
        self.initial_mp = self.get_state(self.symbol[0])[self.n_time_step - 1]
        self.initial_mp_2 = self.get_state(self.symbol[1])[self.n_time_step - 1]
        self.initial_inv_pnl = 0
        self.initial_inv_pnl_2 = 0


        while (self.trader.get_last_price(self.symbol[0]) == 0) or (self.trader.get_last_price(self.symbol[1]) == 0):
            sleep(1)
            print("LT waiting")

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
        # for act in [0, -1, 1]:
        #     action_list.append(self.lp_action(ticker=self.symbol, signal=act))
        #     action_params.append(act)
        # action_space = gym.spaces.Discrete(3)
        for action_1 in [0, -1, 1]:
            for action_2 in [0, -1, 1]:
                action_list.append((self.lp_action(ticker=self.symbol[0], signal=action_1),
                                   self.lp_action(ticker=self.symbol[1], signal=action_2)))
                action_params.append((action_1, action_2))

        action_space = gym.spaces.Discrete(9)
        return action_space, action_list, action_params

    def execute_action(self, action):
        if action == None:
            return None
        else:
            order_ids = []
            place_order = self.action_list[action]
            for place_action in place_order:
                order_ids.append(place_action(self.trader))
            return order_ids

    def check_order(self, order_id):
        # return size and type of filled order
        # if order was not filled, cancel it and return (0, None)
        order = self.trader.get_order(order_id)
        # print(order.status)
        if order.status == shift.Order.Status.FILLED:
            return True
        else:
            return False

    def _data_thread(self,symbol):
        # thread constantly collecting midprice data
        print(f"Data thread starts")

        while self.trader.is_connected() and self.data_thread_alive:
            best_price = self.trader.get_best_price(symbol)
            best_bid = best_price.get_bid_price()
            best_ask = best_price.get_ask_price()
            if symbol == self.symbol[0]:
                if (best_bid == 0) and (best_ask == 0):
                    if len(self.midprice_list) > 0:
                        self.midprice_list.append(self.midprice_list[-1])
                elif (best_bid == 0) or (best_ask == 0):
                    self.midprice_list.append(max(best_bid, best_ask))
                else:
                    self.midprice_list.append((best_bid + best_ask) / 2)
            elif symbol == self.symbol[1]:
                if (best_bid == 0) and (best_ask == 0):
                    if len(self.midprice_list) > 0:
                        self.midprice_list_2.append(self.midprice_list_2[-1])
                elif (best_bid == 0) or (best_ask == 0):
                    self.midprice_list_2.append(max(best_bid, best_ask))
                else:
                    self.midprice_list_2.append((best_bid + best_ask) / 2)

            sleep(self.time_step)

        print("Data Thread stopped.")

    def get_state(self, symbol):
        # return a list with these values: mp, sp, inv, bp, bid_volumes, ask_volumes, order_size, target order proportions, gamma, alpha, w

        while True:
            if len(self.midprice_list) == self.n_time_step:
                best_price = self.trader.get_best_price(symbol)
                best_bid = best_price.get_bid_price()
                best_ask = best_price.get_ask_price()
                if (best_bid == 0) or (best_ask == 0):
                    spread = 0
                else:
                    spread = best_ask - best_bid

                inv = self.trader.get_portfolio_item(symbol).get_shares() // 100

                bid_book = self.trader.get_order_book(
                    symbol, shift.OrderBookType.LOCAL_BID, self.order_book_range
                )
                ask_book = self.trader.get_order_book(
                    symbol, shift.OrderBookType.LOCAL_ASK, self.order_book_range
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
                # print("waiting for to collect more data")
                sleep(1)

    def step(self, action):
        act_dir = self.action_params[action]
        # print(act_dir)

        # ACTION: #################################################################################################################
        order_ids = self.execute_action(action)
        if act_dir[0] == -1:
            self.order_queue.append(-1)
            self.sell_count = self.order_queue.count(-1)
        elif act_dir[0] == 1:
            self.order_queue.append(1)
            self.buy_count = self.order_queue.count(1)
        elif act_dir[0] == 0:
            self.order_queue.append(0)
        if act_dir[1] == -1:
            self.order_queue_2.append(-1)
            self.sell_count_2 = self.order_queue_2.count(-1)
        elif act_dir[1] == 1:
            self.order_queue_2.append(1)
            self.buy_count_2 = self.order_queue_2.count(1)
        elif act_dir[1] == 0:
            self.order_queue_2.append(0)
        self.steps_elapsed += 1

        sleep(self.step_time)

        # STATE: ##################################################################################################################
        offset = self.n_time_step - 1
        #symbol 1__________________________________________________________________
        state = self.get_state(self.symbol[0])
        if( order_ids != None) and (order_ids != []) and (order_ids[0] != None):
            if self.check_order(order_ids[0]):
                self.stats["filled"].append(True)
            else:
                self.stats["filled"].append(False)
        else:   self.stats["filled"].append(True)

        curr_mp = state[0 + offset]
        curr_inv = state[2 + offset]
        inv_pnl = abs((curr_inv * (curr_mp - self.initial_mp)) - self.initial_inv_pnl)
        self.initial_mp = curr_mp
        self.initial_inv_pnl += curr_inv * (curr_mp - self.initial_mp)
        curr_q = (
            abs(self.targ_buy_frac - (self.buy_count / len(self.order_queue)))
            + abs(self.targ_sell_frac - (self.sell_count / len(self.order_queue)))
        ) / 2
        delta_q = curr_q - self.prev_q
        self.prev_q = curr_q

        #symbol 2___________________________________________________________________
        state_2 = self.get_state(self.symbol[1])
        if( order_ids != None) and (order_ids != []) and (order_ids[1] != None):
            if self.check_order(order_ids[1]):
                self.stats["filled_2"].append(True)
            else:
                self.stats["filled_2"].append(False)
        else:   self.stats["filled_2"].append(True)

        curr_mp_2 = state_2[0 + offset]
        curr_inv_2 = state[2 + offset]
        inv_pnl_2 = abs((curr_inv_2 * (curr_mp_2 - self.initial_mp_2)) - self.initial_inv_pnl_2)
        self.initial_mp_2 = curr_mp_2
        self.initial_inv_pnl_2 += curr_inv_2 * (curr_mp_2 - self.initial_mp_2)

        
        curr_q_2 = (
            abs(self.targ_buy_frac - (self.buy_count_2 / len(self.order_queue_2)))
            + abs(self.targ_sell_frac - (self.sell_count_2 / len(self.order_queue_2)))
        ) / 2
        delta_q_2 = curr_q_2 - self.prev_q_2
        self.prev_q_2= curr_q_2

        #Reward ################################################
        curr_pnl = self.trader.get_portfolio_summary().get_total_realized_pl()# + self.trader.get_unrealized_pl(symbol=self.symbol)
        pnl = curr_pnl - self.initial_pnl
        self.initial_pnl = curr_pnl
        total_inv_pnl = inv_pnl + inv_pnl_2
        total_pnl = pnl - (self.gamma * total_inv_pnl)
        avg_delta_q = (delta_q + delta_q_2)/2
        reward = (self.w * self.alpha * total_pnl) - ((1 - self.w) * avg_delta_q)
        # print(f"{self.symbol} Reward: {reward}")

        # end conditions
        done = False
        if self.max_iters == None:
            pass
        elif self.steps_elapsed > self.max_iters:
            done = True

        self.stats["act_dir"].append(act_dir[0])
        self.stats["rew"].append(reward)
        self.stats["total_pnl"].append(total_pnl)
        self.stats["pnl"].append(pnl)
        self.stats["inv_pnl"].append(inv_pnl)
        self.stats["mp"].append(curr_mp)
        self.stats["sp"].append(state[1 + offset])
        self.stats["inv"].append(curr_inv)
        self.stats["bp"].append(self.trader.get_portfolio_summary().get_total_bp())
        self.stats["bid_vol"].append(sum(state[(5 + offset) : (6 + offset + self.order_book_range)]) )
        self.stats["ask_vol"].append(sum(state[(5 + offset + self.order_book_range) :]))
        self.stats["buy_frac"].append(self.buy_count / len(self.order_queue))
        self.stats["sell_frac"].append(self.sell_count / len(self.order_queue))
        self.stats["curr_q"].append(curr_q)

        self.stats["act_dir_2"].append(act_dir[1])
        self.stats["inv_pnl_2"].append(inv_pnl_2)
        self.stats["mp_2"].append(curr_mp_2)
        self.stats["sp_2"].append(state_2[1 + offset])
        self.stats["inv_2"].append(curr_inv_2)
        self.stats["bid_vol_2"].append(sum(state_2[(5 + offset) : (6 + offset + self.order_book_range)]) )
        self.stats["ask_vol_2"].append(sum(state_2[(5 + offset + self.order_book_range) :]))
        self.stats["buy_frac_2"].append(self.buy_count_2 / len(self.order_queue_2))
        self.stats["sell_frac_2"].append(self.sell_count_2 / len(self.order_queue_2))
        self.stats["curr_q_2"].append(curr_q_2)

        if self.switch_steps != 0:
            if self.steps_elapsed % self.switch_steps == 0:
                index = int(self.steps_elapsed / self.switch_steps)
                self.targ_buy_frac = self.target_buy_sell_flows[0][index % len(self.target_buy_sell_flows[0])]
                self.targ_sell_frac = self.target_buy_sell_flows[1][index % len(self.target_buy_sell_flows[1])]
        
        state_final = np.concatenate((state, state_2[:len(state_2)-5]), axis=0)
        
        return state_final, reward, done, False, dict()

    def close_positions(self):
        # close all positions for given ticker
        print("running close positions function for", self.symbol)

        # close any long positions
        item = self.trader.get_portfolio_item(self.symbol[0])
        long_shares = item.get_long_shares()
        if long_shares > 0:
            print(f"{self.symbol} market selling because long shares = {long_shares}")
            # order = shift.Order(shift.Order.Type.MARKET_SELL, self.symbol, long_shares)
            # self.trader.submit_order(order)
            # sleep(0.2)
            rejections = 0
            while item.get_long_shares() > 0:
                order = shift.Order(
                    shift.Order.Type.MARKET_SELL, self.symbol, long_shares
                )
                self.trader.submit_order(order)
                sleep(2)
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
        item = self.trader.get_portfolio_item(self.symbol[0])
        short_shares = item.get_short_shares()
        if short_shares > 0:
            print(f"{self.symbol} market buying because short shares = {short_shares}")
            # order = shift.Order(shift.Order.Type.MARKET_BUY, self.symbol, long_shares)
            # self.trader.submit_order(order)
            # sleep(0.2)
            rejections = 0
            while item.get_short_shares() > 0:
                order = shift.Order(
                    shift.Order.Type.MARKET_BUY, self.symbol, short_shares
                )
                self.trader.submit_order(order)
                sleep(2)
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
        self.trader.cancel_all_pending_orders()
        self.close_positions()
        self.buy_count = 0
        self.sell_count = 0
        self.steps_elapsed = 0
        state = self.get_state(self.symbol[0])
        state_2 = self.get_state(self.symbol[0])
        state_final = np.concatenate((state, state_2[:len(state_2)-5]), axis=0)
        return state_final, dict()

    def hard_reset(self):
        self.reset()
        for key in self.stats.keys():
            self.stats[key] = []

    def kill_thread(self):
        self.data_thread_alive = False

    def save_to_csv(self, epoch):
        df = pd.DataFrame.from_dict(self.stats)
        df.to_csv(epoch, index=False)

    def _getCurrentPosition(self):
        return int(self.trader.getPortfolioItem(self.symbol).getShares())

    def __del__(self):
        self.kill_thread()

    def __call__(self, *args, **kwds):
        return self
