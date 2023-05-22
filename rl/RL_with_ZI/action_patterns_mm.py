import shift
import numpy as np
from time import sleep

class Action_Patterns:
    # def __init__(self) -> None:
    #     pass
    
    def generate_mm_trader(self, 
                           *,
                           ticker: str,
                           order_size_ratio: float,     # this times average number of orders on each price level = the size of the order for bid or ask
                           symmetric_e = float,
                            asymmetric_e = float,
                            hedge_ratio = float,
                           tick_size = 0.01,
                           book_depth = 5):

        def func(order_list: list, trader):

            # check length of order_list
            # if len(order_list) > max_order_list_length:
            #     print(f"warning, order list length reaches max {len(order_list)}/{max_order_list_length}.")
            #     return


            # get necessary data
            bp = trader.get_best_price(ticker)
            bid = bp.get_bid_price()
            ask = bp.get_ask_price()
            spread = ask - bid
            #print("spread:", spread)

            ask_book = trader.get_order_book(ticker, shift.OrderBookType.LOCAL_ASK, book_depth)
            bid_book = trader.get_order_book(ticker, shift.OrderBookType.LOCAL_BID, book_depth)

            # get data
            best_bid_size = 0
            best_ask_size = 0
            for order in ask_book:
                best_ask_size = order.size
            for order in bid_book:
                best_bid_size = order.size
            total_size = best_bid_size + best_ask_size
            #print(f"total_size:{total_size}")

            # setting order sizes
            bid_size = int(order_size_ratio * total_size / (book_depth*2))
            ask_size = int(order_size_ratio * total_size / (book_depth*2))
            
            # adjust order sizes depending on current holdings
            item = trader.get_portfolio_item(ticker)
            imbalance = int(item.get_long_shares()/100) - int(item.get_short_shares()/100)      #inventory imbalance = long - short
            if (imbalance != 0):
                if imbalance > 0:
                    if imbalance > 10: imbalance = 10
                    ask_size += abs(imbalance)
                else:
                    if imbalance < 10: imbalance = 10
                    bid_size += abs(imbalance)
                

            #print(f'bid order size: {bid_size}')



            # make sure the spread is large enough to profit
            if (spread >= (tick_size * 3)):
                # submit orders
                if bid_size > 0:
                    limit_buy = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, bid_size, bid + tick_size)
                    order_list.append(limit_buy.id)
                    trader.submit_order(limit_buy)
                    #print("bid size:", bid_size)
                if ask_size > 0:
                    limit_sell = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, ask_size, ask - tick_size)
                    order_list.append(limit_sell.id)
                    trader.submit_order(limit_sell)
                    #print("ask size:", ask_size)

        return func
                    

