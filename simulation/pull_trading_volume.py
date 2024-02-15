def pull_trade_records(*, con, ticker):

    query = f"""select session_id as session_id, real_time as real_time, execution_time as execution_time,
            symbol as symbol, price as price, size as size, trader_id_1 as trader_id_1, trader_id_2 as trader_id_2,
            order_id_1 as order_id_1, order_id_2 as order_id_2, order_type_1 as order_type_1,
            order_type_2 as order_type_2, time_1 as time_1, time_2 as time_2, decision as decision,
            destination as destination from trading_records where symbol='CS{ticker}'"""
    portfolio_query = f"""select * from portfolio_summary where total_pl != 0"""
    trade_recs = pd.read_sql_query(query, con)
    portfolio_summary = pd.read_sql_query(portfolio_query, con)

    # trade_recs = trade_recs[trade_recs["decision"] == "2"]
    # trade_recs = trade_recs[trade_recs["destination"] == "SHIFT"]
    # trade_recs['execution_time'] = pd.to_datetime(trade_recs['execution_time'])
    # trade_recs = trade_recs.set_index("execution_time")
    # trade_recs["sum_over_time"] = trade_recs["size"].rolling("2s").sum()
    if ticker == 2:
        trade_recs.to_csv(f"/home/shiftpub/Results_Simulation{ticker}/iteration_info/trade_rec.csv")
        portfolio_summary.to_csv(f"/home/shiftpub/Results_Simulation{ticker}/iteration_info/port_sum.csv")
    else:
        trade_recs.to_csv(f"/home/shiftpub/Results_Simulation/iteration_info/trade_rec.csv")
        portfolio_summary.to_csv(f"/home/shiftpub/Results_Simulation/iteration_info/port_sum.csv")
    # print([ele for ele in res])


if __name__ == "__main__":
    import sqlalchemy
    import sys
    sys.path.insert(1, "/home/shiftpub")
    from db_config import db_string
    db = sqlalchemy.create_engine(db_string)
    import pandas as pd
    with db.connect() as con: pull_trade_records(con = con, ticker = 1)
     