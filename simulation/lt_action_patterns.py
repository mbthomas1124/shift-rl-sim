import shift

class Action_Patterns:    

    def generate_momentum_trader(self,
                        ticker,
                        ratio_thres,
                        window,
                        order_size,
                        levels = 5
                        ):
        def func(trader, table):
            order_id = None
            bid_volume = 0
            ask_volume = 0
            for i in range(1, window+1):
                data = table[-i]
                bid_volume += data[0]
                ask_volume += data[1]
            bid_volume /= window
            ask_volume /= window
                
            total_volume = bid_volume + ask_volume
            signal = (bid_volume - ask_volume) / total_volume

            if (signal >= ratio_thres):
                # buy
                if trader.get_portfolio_summary().get_total_bp() > (order_size * 100 * trader.get_last_price(ticker)):
                    order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, order_size) 
                    trader.submit_order(order)
                    order_id = order.id
                else:  
                    print(f"{ticker} insufficient buying power: {trader.get_portfolio_summary().get_total_bp()}")
            
            elif (signal <= (-1 * ratio_thres)):
                # sell
                long_shares = trader.get_portfolio_item(ticker).get_long_shares() > 0
                if long_shares < order_size:
                    # involves short selling
                    short_size = order_size - long_shares
                    if trader.get_portfolio_summary().get_total_bp() > (short_size * 2 * 100 * trader.get_last_price(ticker)):
                        order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, order_size) 
                        trader.submit_order(order)
                        order_id = order.id
                    else:  
                        print(f"{ticker} insufficient buying power: {trader.get_portfolio_summary().get_total_bp()}")
                else:
                    order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, order_size) 
                    trader.submit_order(order)
                    order_id = order.id
                    
            return order_id, signal
        
        return func


    def generate_momentum_trader2(self,
                        ticker,
                        signal,
                        order_size,
                        levels = 5
                        ):
        def func(trader, table):
            order_id = None

            if (signal == 1):
                # buy
                if trader.get_portfolio_summary().get_total_bp() > (order_size * 100 * trader.get_last_price(ticker)):
                    order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, order_size) 
                    trader.submit_order(order)
                    order_id = order.id
                else:  
                    print(f"{ticker} insufficient buying power: {trader.get_portfolio_summary().get_total_bp()}")
            
            elif (signal == -1):
                # sell
                long_shares = trader.get_portfolio_item(ticker).get_long_shares() > 0
                if long_shares < order_size:
                    # involves short selling
                    short_size = order_size - long_shares
                    if trader.get_portfolio_summary().get_total_bp() > (short_size * 2 * 100 * trader.get_last_price(ticker)):
                        order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, order_size) 
                        trader.submit_order(order)
                        order_id = order.id
                    else:  
                        print(f"{ticker} insufficient buying power: {trader.get_portfolio_summary().get_total_bp()}")
                else:
                    order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, order_size) 
                    trader.submit_order(order)
                    order_id = order.id
                    
            return order_id, signal
        
        return func
