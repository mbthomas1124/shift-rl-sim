import argparse
import os
import pprint

import gym
import numpy as np
import datetime as dt
from time import sleep
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import DQNPolicy, RainbowPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

from lt_rl_env import SHIFT_env
import shift


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--step-per-epoch', type=int, default=50)
    parser.add_argument('--step-per-collect', type=int, default=20)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
    )
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    args = parser.parse_known_args()[0]
    return args


def test_dqn(args=get_args()):
    with shift.Trader("test001") as trader1, shift.Trader("test002") as trader2:
        trader1.connect("initiator.cfg", "password")
        trader2.connect("initiator.cfg", "password")
        trader1.sub_all_order_book()
        trader2.sub_all_order_book()
        sleep(5)

        # wait for starting time
        start_time = dt.datetime.now().replace(minute = 15, second = 0)
        while dt.datetime.now() < start_time:
            sleep(1)
        print("Starting DQN")
        
        env_list = []
        env_list.append(SHIFT_env(trader = trader1,
                symbol= 'CS1'))
        # env_list.append(SHIFT_env(trader = trader2,
        #         symbol= 'CS2'))

        args.state_shape = env_list[0].observation_space.shape or env_list[0].observation_space.n
        args.action_shape = env_list[0].action_space.shape or env_list[0].action_space.n

        # seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)

        optim = torch.optim.Adam(net.parameters(), lr=args.lr)

        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq,
        )

        # load saved policy
        log_path = os.path.join(args.logdir, 'Liquidity-Taker', 'dqn')
        # policy.load_state_dict(torch.load(os.path.join(log_path, 'policy.pth')))
        
        # buffer
        buff = VectorReplayBuffer(args.buffer_size, buffer_num=1)
        
        # collector
        collector = Collector(
            policy,
            DummyVectorEnv(env_list),
            buff,
            exploration_noise=True
        )

        print("\nCOLLECTING BEGINS NOW\n")
        collector.collect(n_step=args.batch_size * args.training_num)
        print("\nCOLLECTING HAS CONCLUDED\n")
        
        # log
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)

        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

        def stop_fn(mean_rewards):
            return False

        def train_fn(epoch, env_step):
            # eps annnealing, just a demo
            if env_step <= 10:
                policy.set_eps(args.eps_train)
            elif env_step <= 50:
                eps = args.eps_train - (env_step - 10) / \
                    40 * (0.9 * args.eps_train)
                policy.set_eps(eps)
            else:
                policy.set_eps(0.1 * args.eps_train)

        def test_fn(epoch, env_step):
            policy.set_eps(args.eps_test)

        # trainer
        print("\nTRAINING BEGINS NOW\n")
        result = offpolicy_trainer(
            policy,
            collector,
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
            logger=logger,
        )
        print("\nTRAINING HAS CONCLUDED\n")


        if __name__ == '__main__':    
            # for env in env_list:
            #     env.hard_reset()
            # sleep(5)

            # Let's watch its performance!
            print("\nLAST RUN\n")
            policy.eval()
            collector.collect(n_step=args.step_per_epoch)

        for env in env_list:
            env.save_env()

        for env in env_list:
            env.reset()


if __name__ == '__main__':
    test_dqn(get_args())