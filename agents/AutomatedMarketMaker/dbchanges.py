import psycopg2 as c

vms = [
"155.246.104.85",
] 

'''
For each user that has firstname 'mm' for marketmaker they will be given 1000000 shares of CS1-CS10 and buying power of 100000000
For each user that has firstname 'Test' for liquidity demander they will be given 10000 shares of CS1-CS10 and buying power of 1000000
'''
for vm in vms:
    conn = c.connect(
        host=vm,
        database="shift_brokeragecenter",
        user="hanlonpgsql4",
        password="XIfyqPqM446M")
    
    cur = conn.cursor()
        
    cur.execute('delete from portfolio_items')
    cur.execute('delete from portfolio_summary')
    cur.execute('delete from trading_records')

    cur.execute('select id from traders where firstname=\'mm\'')
    mmIds = list(map(lambda t : t[0], cur.fetchall()))
    cur.execute('select id from traders where firstname=\'Test\'')
    ldIds = list(map(lambda t : t[0], cur.fetchall()))
    
    for mm in mmIds:
        for i in range(1,11):
            cur.execute(f'insert into portfolio_items (id, symbol, long_price, long_shares) values (\'{mm}\', \'CS{i}\', 100, 1000000)')
        cur.execute(f'insert into portfolio_summary (id, buying_power, holding_balance, borrowed_balance, total_pl, total_shares) values (\'{mm}\', 100000000, 0, 0, 0, {1000000*10})')
        conn.commit()

    for ld in ldIds:
        for i in range(1,11):
            cur.execute(f'insert into portfolio_items (id, symbol, long_price, long_shares) values (\'{ld}\', \'CS{i}\', 100, 10000)')
        cur.execute(f'insert into portfolio_summary (id, buying_power, holding_balance, borrowed_balance, total_pl, total_shares) values (\'{ld}\', 1000000, 0, 0, 0, {10000*10})')
        conn.commit()

    conn.close()
