import argparse
import os
import pprint
import datetime as dt
import pandas as pd

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from time import sleep
from datetime import timedelta, datetime
from threading import Thread

from tianshou.data import Collector, VectorReplayBuffer, AsyncCollector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import DQNPolicy, PPOPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from F_RL_env_mm import SHIFT_env as Env
import shift


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str, default='Acrobot-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.1)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=200)
    parser.add_argument('--step-per-collect', type=int, default=20)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64])
    parser.add_argument(
        '--dueling-q-hidden-sizes', type=int, nargs='*', default=[64, 64]
    )
    parser.add_argument(
        '--dueling-v-hidden-sizes', type=int, nargs='*', default=[64, 64]
    )
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.02)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return parser.parse_args()

def LOB_record(trader, trail):
    check_frequency = 0.5
    print("START")

    num_tickers = 1

    tickers = []
    for i in range(num_tickers):
        tickers.append({"sp":[], "mid":[], "ask_5":[],"ask_4":[],"ask_3":[],"ask_2":[],"ask_1":[], "bid_1":[], "bid_2":[], "bid_3":[], "bid_4":[], "bid_5":[]})

    while True:
        #get price
        for i in range(num_tickers):
            tick = 'CS'+str(i+1)
            best_p = trader.get_best_price(tick)
            sp = round((best_p.get_ask_price() - best_p.get_bid_price()),4)
            mid = round(((best_p.get_ask_price()+best_p.get_bid_price())/2),4)
            tickers[i]["sp"].append(sp)
            tickers[i]["mid"].append(mid)
            ask_book = trader.get_order_book(tick, shift.OrderBookType.LOCAL_ASK, 5)
            bid_book = trader.get_order_book(tick, shift.OrderBookType.LOCAL_BID, 5)
            # get data
            bid_size = []
            ask_size = []
            for order in ask_book:
                ask_size.append(order.size) 
            for order in bid_book:
                bid_size.append(order.size)

            while len(ask_size) < 5:
                ask_size.append(0)
            while len(bid_size) < 5:
                bid_size.append(0)
            #print("bid ask volume:", bid_size_1, ask_size_1)

            tickers[i]["ask_1"].append(ask_size[0])
            tickers[i]["ask_2"].append(ask_size[1])
            tickers[i]["ask_3"].append(ask_size[2])
            tickers[i]["ask_4"].append(ask_size[3])
            tickers[i]["ask_5"].append(ask_size[4])
            tickers[i]["bid_1"].append(bid_size[0])
            tickers[i]["bid_2"].append(bid_size[1])
            tickers[i]["bid_3"].append(bid_size[2])
            tickers[i]["bid_4"].append(bid_size[3])
            tickers[i]["bid_5"].append(bid_size[4])
            
        sleep(check_frequency)
        #check for stopping time
        global stop_threads
        if stop_threads:
            # save results
            for i in range(num_tickers):
                file_path = f"./iteration_info/Orderbook_trial{trail}_CS{i+1}.csv"
                pd.DataFrame(tickers[i]).to_csv(file_path, index=False)
            print("Recording Thread DONE")
            break
        
stop_threads = False

def test_dqn(args=get_args()):
    trader = shift.Trader("marketmaker")
    #trader2 = shift.Trader("test007")

    trader.disconnect()
    #trader2.disconnect()
    policy_name = "temp"
    test_steps = 200
    try:
        trader.connect("initiator.cfg", "password")
        #trader2.connect("initiator.cfg", "password")
        trader.sub_all_order_book()
        #trader2.sub_all_order_book()
        #wait for starting time
        sleep(2)
        print("bp:",trader.get_portfolio_summary().get_total_bp())

        # start_time = datetime.now().replace(minute = 15, second = 0)
        # while datetime.now() < start_time:
        #     sleep(1)
        print("Starting")
        
        #natural stop:
        natural_s = 100
        
        #initiate envs
        env = Env(trader = trader,
            rl_t = 3,
            info_step= 0.25,
            nTimeStep=10,
            ODBK_range=5,
            symbol='CS1',
            max_order_list_length=10)
        
        """env2 = Env(trader = trader2,
            rl_t = 2,
            strat_t = 0.4,
            nTimeStep=5,
            ODBK_range=5,
            symbol='CS2',
            max_order_list_length=10)"""
    
        env_list = [env]
        
        args.state_shape = env.observation_space.shape or env.observation_space.n
        args.action_shape = env.action_space.shape or env.action_space.n
        print(f"action space: {args.action_shape}")

        # seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # train_envs.seed(args.seed)
        # test_envs.seed(args.seed)
        # model
        Q_param = {"hidden_sizes": args.dueling_q_hidden_sizes}
        V_param = {"hidden_sizes": args.dueling_v_hidden_sizes}
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            dueling_param=(Q_param, V_param)
        ).to(args.device)
        
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        print("optim")
        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq
        )
        log_path = os.path.join(args.logdir, 'dqn')
        #policy.load_state_dict(torch.load(os.path.join(log_path, f'policy_{policy_name}.pth')))
        print("Policy")
        
        # collector
        train_collector = Collector(
            policy,
            DummyVectorEnv(env_list),#, wait_num = 2
            VectorReplayBuffer(args.buffer_size, 1),
            exploration_noise=True
        )
        
        print("train collect")
        # test_collector = Collector(
        #     policy,
        #     DummyVectorEnv([env]),
        #     VectorReplayBuffer(args.buffer_size, 1),
        #     exploration_noise=True
        # )
        # print("test collect")
        # test_collector = Collector(policy, test_envs, exploration_noise=True)
        # policy.set_eps(1)
        train_collector.collect(n_step=args.batch_size * args.training_num)
        #test_collector.collect(n_step=args.batch_size * args.test_num)
        print("finish collect")
        
        #training#################################
        # log
        log_path = os.path.join(args.logdir, 'dqn') #, args.task
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)

        def save_best_fn(policy, name = None):
            print("Policy saved")
            torch.save(policy.state_dict(), os.path.join(log_path, f"policy_{name}.pth"))

        def stop_fn(mean_rewards):
            print("mean rewards:", mean_rewards)
            return mean_rewards >= env.reward_threshold

        def train_fn(epoch, env_step):
            if env_step <= 100000:
                policy.set_eps(args.eps_train) 
            elif env_step <= 500000:
                eps = args.eps_train - (env_step - 100000) / \
                    400000 * (0.5 * args.eps_train)
                policy.set_eps(eps)
            else:
                policy.set_eps(0.5 * args.eps_train)

        def test_fn(epoch, env_step):
            policy.set_eps(args.eps_test)

        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            None,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            update_per_step=args.update_per_step,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger
        )
        print("training done")
        save_best_fn(policy, policy_name)
        #assert stop_fn(result['best_reward'])
        
        if __name__ == '__main__':
            #pprint.pprint(result)
            
            # Let's watch its performance!
            #policy.eval()
            #policy.set_eps(args.eps_test)
            # test_envs.seed(args.seed)
            # test_collector.reset()
            process = Thread(target=LOB_record, args=(trader, 1))
            process.start()
            env.isSave = True
            #env2.isSave = True
            sleep(1)
            
            print("start colleting result")
            result = train_collector.collect(n_step = test_steps)#n_episode=1
            #print(result)
            #rews, lens = result["rews"], result["lens"]
            #print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
            global stop_threads
            stop_threads = True
            process.join()
            
            
            
    except KeyboardInterrupt:
        trader.disconnect()
        #trader2.disconnect()
      
    finally:
        env.save_to_csv("new_mm")
        #env2.save_to_csv(2)
        trader.disconnect()
        #trader2.disconnect()



if __name__ == '__main__':
    test_dqn(get_args())
