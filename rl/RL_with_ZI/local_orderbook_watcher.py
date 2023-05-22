import shift
import argparse
import time

def main(args):
    trader = shift.Trader("democlient")
    trader.disconnect()
    trader.connect("initiator.cfg", "password")
    trader.sub_all_order_book()
    try:
        while trader.is_connected():

            #last_price = self.trader.getLastPrice(self.symbol)

            Ask_ls = trader.get_order_book(args.ticker, shift.OrderBookType.LOCAL_ASK, args.depth)
            # assert Ask_ls, f'getOrderBook: return empty list: {self.symbol}-ASK-{self.ODBK_range}'
            Ask_ls += [None]*(args.depth - len(Ask_ls))

            Bid_ls = trader.get_order_book(args.ticker, shift.OrderBookType.LOCAL_BID, args.depth)
            # assert Bid_ls, f'getOrderBook: return empty list: {self.symbol}-BID-{self.ODBK_range}'
            Bid_ls += [None]*(args.depth - len(Bid_ls))
            #get remaining buying power
            
            for i in range(args.depth):
                bid_order, ask_order = Bid_ls[i], Ask_ls[i]
                if bid_order is not None:
                    bid_string = f"| BP {bid_order.price:0.2f}, BQ {bid_order.size} "
                else:
                    bid_string = f"| BP None, BQ None "
                    
                if ask_order is not None:
                    ask_string = f"| AP {ask_order.price:0.2f}, AQ {ask_order.size} "
                else:
                    ask_string = f"| AP None, AQ None "
                print(f"Ticker: {args.ticker} | Level {i+1} "+bid_string+ask_string)
            time.sleep(args.frequency)
            print()
            
            # bp = trader.get_portfolio_summary().get_total_bp()

            #get best bid and ask prices
            # best_p = trader.get_best_price(args.ticker)

            # info = self.LOB_to_list(Ask_ls, Bid_ls, best_p, bp)
            
    finally:
        trader.disconnect()

    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str,
                        help='the ticker to watch')
    parser.add_argument('--depth', default=20, type=int,
                        help='orderbook depth')
    parser.add_argument('--frequency', default=1., type=float,
                        help='sample frequency')
    # parser.add_argument('--work_directory', type=str,
    #                     help='directory to save models')
    

    args = parser.parse_args()
    main(args)