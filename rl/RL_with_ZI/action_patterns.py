import shift
import numpy as np
import time

class Action_Patterns:
    # def __init__(self) -> None:
    #     pass
    
    def generate_ZI_trader(self, 
                           *,
                           ticker: str,
                           initial_price: float,
                           initial_volatility: float,
                           max_order_size: int,
                           order_number: int, 
                           lambda_: float, # range from 0 to 1
                           # sleep_time: float,
                           max_order_list_length = 20, 
                           trade_percent = 0.05):
        def func(order_list: list, trader, center_price):

            # check length of order_list
            if len(order_list) > max_order_list_length:
                print(f"warning, order list length reaches max {len(order_list)}/{max_order_list_length}.")
                return
            
            new_orders = []
            for _ in range(order_number):
                last_price = trader.get_last_price(ticker)
                last_price = last_price if last_price > 0. else initial_price
                if np.random.uniform() > 0.5:
                    best_bid = trader.get_best_price(ticker).get_bid_price()
                    best_bid = best_bid if best_bid > 0. else last_price
                    target_price = min(best_bid, last_price)
                    target_price = lambda_ * center_price + (1 - lambda_) * target_price
                    order_price = np.round(target_price + initial_volatility * np.random.normal(), 2)
                    order_size = min(max_order_size, 
                                        np.floor(trade_percent * trader.get_portfolio_summary().get_total_bp() / (100 * order_price)) )
                    order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, int(order_size), order_price) 
                    if order.size > 0:
                        order_list.append(order.id)
                        new_orders.append(order.id)
                        trader.submit_order(order)
                    else:  
                        print(f"insufficient buying power: {trader.get_portfolio_summary().get_total_bp()}")
                else:
                    best_ask = trader.get_best_price(ticker).get_ask_price()
                    best_ask = best_ask if best_ask > 0. else last_price
                    target_price = max(best_ask, last_price)
                    target_price = lambda_ * center_price + (1 - lambda_) * target_price
                    order_price = np.round(target_price + initial_volatility * np.random.normal(), 2)
                    order_size = min(max_order_size, 
                                        np.floor(trade_percent * trader.get_portfolio_summary().get_total_bp() / (100 * order_price)) )
                    order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, int(order_size), order_price)
                    if order.size > 0:
                        order_list.append(order.id)
                        new_orders.append(order.id)
                        trader.submit_order(order)  
                    else:
                        print(f"insufficient buying power: {trader.get_portfolio_summary().get_total_bp()}")
            print(f"new orders placed: {len(new_orders)}")
        return func
                    
