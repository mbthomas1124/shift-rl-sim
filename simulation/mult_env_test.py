from RL_env import SHIFT_env as Env
import shift
import time
import random
import sys
DEBUG = True
if DEBUG:
    from pathlib import Path
    import os
    print(os.chdir(Path(__file__).parent))


action = int(sys.argv[1])
lam = float(sys.argv[2])
ordersize = int(sys.argv[3])
orderlen = int(sys.argv[4])
testtime = int(sys.argv[5])



num_tickers = 2

trader = shift.Trader("marketmaker")
trader.disconnect()
trader.connect("initiator.cfg", "password")
trader.sub_all_order_book()

ZI_param = {'max_order_size': ordersize, 
            'order_number': 5,
            'trade_percent': 0.01,
            'lambda': lam}
env_list = []
for i in range(num_tickers):
    env_list.append(Env(trader = trader,
        t = 0.5,
        nTimeStep=10,
        ODBK_range=5,
        symbol= 'CS'+str(i+1),
        target_price=100.5, 
        max_order_list_length=orderlen, 
        ZI_param = ZI_param))

try:
    time.sleep(10)

    for i in range(num_tickers):
        env_list[i].reset()

    print("action begin")
    # 0 -- 0.05, 1 -- 0.1, 2 -- 0.2, 3 -- 0.4
    for i in range(num_tickers):
        env_list[i].step(action)
    
    time.sleep(testtime * 60)

    # buffer
    time.sleep(30)

except KeyboardInterrupt:
    trader.disconnect()
finally:
    trader.disconnect()

print("mult_env_test done")

