import argparse
import os
import pprint
import datetime as dt
import pandas as pd

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal
from time import sleep
from datetime import timedelta, datetime
from threading import Thread

from tianshou.data import Collector, VectorReplayBuffer, AsyncCollector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import DQNPolicy, PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, ActorCritic
from F_RL_env_mm import SHIFT_env as Env
from tianshou.utils.net.continuous import ActorProb, Critic
import shift


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str, default='Acrobot-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.1)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--repeat-per-collect', type=int, default=5)
    
    parser.add_argument('--step-per-collect', type=int, default=5)
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
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)
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
    test_steps = 10
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
        args.max_action = env.action_space.high[0]
        print("args.max_action", args.max_action)
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
            device=args.device#,dueling_param=(Q_param, V_param)
        ).to(args.device)
        actor = ActorProb(
            net, args.action_shape, max_action=args.max_action, device=args.device
        ).to(args.device)
        critic = Critic(
            Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device),
            device=args.device
        ).to(args.device)
        actor_critic = ActorCritic(actor, critic)
        # orthogonal initialization
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
        print("optim")
        
        def dist(*logits):
            return Independent(Normal(*logits), 0)
    
    
        policy = PPOPolicy(
            actor,
            critic,
            optim,
            torch.distributions.Normal,
            discount_factor=args.gamma,
            max_grad_norm=args.max_grad_norm,
            eps_clip=args.eps_clip,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            reward_normalization=args.rew_norm,
            advantage_normalization=args.norm_adv,
            recompute_advantage=args.recompute_adv,
            dual_clip=args.dual_clip,
            value_clip=args.value_clip,
            gae_lambda=args.gae_lambda,
            action_space=env.action_space,
        )
        log_path = os.path.join(args.logdir, 'ppo')
        #policy.load_state_dict(torch.load(os.path.join(log_path, f'policy_{policy_name}.pth')))
        print("Policy")
        
        # collector
        train_collector = Collector(
            policy,
            DummyVectorEnv(env_list),#, wait_num = 2
            VectorReplayBuffer(args.buffer_size, len(env_list)),
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
        #train_collector.collect(n_step=args.batch_size * args.training_num)
        #test_collector.collect(n_step=args.batch_size * args.test_num)
        #print("finish collect")
        
        #training#################################
        # log
        log_path = os.path.join(args.logdir, 'dqn') #, args.task
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)

        def save_best_fn(policy, name = None):
            print("Policy saved")
            torch.save(policy.state_dict(), os.path.join(log_path, f"ppo_policy_{name}.pth"))

        def stop_fn(mean_rewards):
            print("mean rewards:", mean_rewards)
            return mean_rewards >= env.reward_threshold

        def train_fn(epoch, env_step):
            print("train_fn")
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
        result = OnpolicyTrainer(
            policy,
            train_collector,
            test_collector = None,
            max_epoch = args.epoch,
            step_per_epoch = args.step_per_epoch,
            repeat_per_collect = args.repeat_per_collect,
            episode_per_test=args.test_num,
            batch_size = args.batch_size,
            step_per_collect=args.step_per_collect,
            #train_fn=train_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            #resume_from_log=args.resume,
        )
        print("training done")
        save_best_fn(policy, policy_name)
        #assert stop_fn(result['best_reward'])
        
        for epoch, epoch_stat, info in result:
            print(f"Epoch: {epoch}")
            print(epoch_stat)
            print(info)
        
        if __name__ == '__main__':
            pprint.pprint(info)
            
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
