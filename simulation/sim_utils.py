def setup_init_portfolio(*, con, trader: str, portfolio: dict):
    """
    Parameters
    ----------
    connect: db connector
    trader: the username of the trader
    portfolio: something like:
    {
        'cash': 1000000,
        'CS1': 10000,
        'CS2': 10000,
        ...
    }
    """
    # cur = con.cursor()
    query = f"select id from traders where username='{trader}'"
    # print(query)
    res = con.execute(query)
    res = [item[0] for item in res]
    id_ = str(res[0])

    for key in portfolio:
        if key == "cash":
            query = f"""
                delete from portfolio_summary where id='{id_}';
                insert into portfolio_summary (id, buying_power, holding_balance, borrowed_balance, total_pl, total_shares) values ('{id_}', {portfolio[key]}, 0, 0, 0, 0);
            """
        else:
            query = f"""
                delete from portfolio_items where id='{id_}' and symbol='{key}';
                insert into portfolio_items (id, symbol, long_price, long_shares, short_shares) values ('{id_}', '{key}', 100, {portfolio[key]},{portfolio[key]});
            """
        # print(query)
        con.execute(query)
        # print([ele for ele in res])


if __name__ == "__main__":
    import sqlalchemy
    from db_config import db_string

    print(db_string)

    db = sqlalchemy.create_engine(db_string)

    import pandas as pd

    """
    portfolio_itemscd
    portfolio_summary
    """
    with db.connect() as con:
        """for i in range(10):
            setup_init_portfolio(con = con,
                                trader = f"agent00{i}",
                                portfolio = {"cash": 1000000, "CS1": 2000})
        for i in range(10, 100):
            setup_init_portfolio(con = con,
                                trader = f"agent0{i}",
                                portfolio = {"cash": 1000000, "CS1": 2000})
        for i in range(100, 235):
            setup_init_portfolio(con = con,
                                trader = f"agent{i}",
                                portfolio = {"cash": 1000000, "CS1": 2000})"""
        setup_init_portfolio(
            con=con, trader="marketmaker_rl_1", portfolio={"cash": 10000000000}
        )  # , "CS1": 200000000      #1 billion for comp_rl_mm
        setup_init_portfolio(
            con=con, trader="marketmaker_rl_2", portfolio={"cash": 10000000000}
        )  # , "CS1": 200000000
        setup_init_portfolio(
            con=con, trader="liquiditytaker_rl_01", portfolio={"cash": 100000000000}
        )
        setup_init_portfolio(
            con=con, trader="liquiditytaker_rl_02", portfolio={"cash": 100000000000}
        )
        setup_init_portfolio(
            con=con, trader="liquiditytaker_rl_03", portfolio={"cash": 100000000000}
        )
        setup_init_portfolio(
            con=con, trader="liquiditytaker_rl_04", portfolio={"cash": 100000000000}
        )
        setup_init_portfolio(
            con=con, trader="liquiditytaker_rl_05", portfolio={"cash": 100000000000}
        )
        setup_init_portfolio(
            con=con, trader="liquiditytaker_rl_06", portfolio={"cash": 100000000000}
        )
        setup_init_portfolio(
            con=con, trader="liquiditytaker_rl_07", portfolio={"cash": 100000000000}
        )
        setup_init_portfolio(
            con=con, trader="liquiditytaker_rl_08", portfolio={"cash": 100000000000}
        )
        setup_init_portfolio(
            con=con, trader="liquiditytaker_rl_09", portfolio={"cash": 100000000000}
        )
        setup_init_portfolio(
            con=con, trader="liquiditytaker_rl_10", portfolio={"cash": 100000000000}
        )
