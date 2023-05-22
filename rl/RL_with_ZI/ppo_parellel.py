"""PPO but trying the parellel subprocess function"""
import argparse
import os
import pprint
from threading import Thread
import gym
import numpy as np
import torch
from torch.distributions import Independent, Normal
#from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer, AsyncCollector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
#from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from F_RL_env_mm import SHIFT_env as Env
from tianshou.utils.net.discrete import Actor as discrete_Actor, Critic as discrete_Critic
#from shared_policy import shared_env_mm as shared_Env
from time import sleep
import shift
from lt_rl_env import SHIFT_env as lt_env

#TODO
#record state action reward next state
#make hidden-size
#research mean field game with shared policy
#mm and lt run
#Sampling the identifiers in a distrbiution so it chagnes while training

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--task', type=str, default='Pendulum-v1')
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--episode-per-collect', type=int, default=20)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64]) #make larger
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='Policies')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--eps-clip', type=float, default=0.3)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.9)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)
    args = parser.parse_known_args()[0]
    return args


def test_ppo(trader, trader_id, w, agent_type, ticker, args=get_args()):
    test_steps = 30

    try:
        #natural stop:
        natural_s = 10
        #########################Market Maker################################
        if agent_type == "mm":
            #initiate envs
            env = Env(trader = trader,
                rl_t = 1,
                info_step= 0.25,
                nTimeStep=10,
                ODBK_range=5,
                symbol=ticker,
                max_order_list_length=20,
                weight = w,
                tar_m = 0.6)
            
            args.state_shape = env.observation_space.shape or env.observation_space.n
            args.action_shape = env.action_space.shape or env.action_space.n
            args.max_action = env.action_space.high[0]
            print(f"state_shape{args.state_shape}")
            print(f"action_shape{args.action_shape}")
            train_envs = [env]

            net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
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

            # replace DiagGuassian with Independent(Normal) which is equivalent
            # pass *logits to be consistent with policy.forward
            def dist(*logits):
                #print(*logits)
                #print(Independent(Normal(*logits), 1))
                return Independent(Normal(*logits), 1)
        #########################Liqudity Taker################################
        elif agent_type == "lt":
            # initiate envs
            env = lt_env(
                trader=trader,
                symbol=ticker,
                step_time=1,
                order_size=10,
                target_buy_frac=w,
                target_sell_frac=1-w,
                risk_aversion=0.5,
                pnl_weighting=0.5,
                normalizer=0.01,
                order_book_range=5,
                max_iterations=None,
            )
            print(w, 1-w)
            train_envs = [env]
            args.state_shape = env.observation_space.shape or env.observation_space.n
            args.action_shape = env.action_space.shape or env.action_space.n
            print(f"state shape: {args.state_shape}")
            print(f"action shape: {args.action_shape}")
            # model
            net = Net(
                args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device,
                softmax=True, norm_layer=torch.nn.InstanceNorm1d, #norm_args=args.state_shape,
            )
            actor = discrete_Actor(net, args.action_shape, device=args.device).to(args.device)
            critic = discrete_Critic(
                net,
                device=args.device
            ).to(args.device)
            actor_critic = ActorCritic(actor, critic)
            # orthogonal initialization
            for m in actor_critic.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

            dist = torch.distributions.Categorical
        

        policy = PPOPolicy(
            actor,
            critic,
            optim,
            dist, #torch.distributions.Normal, 
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
        train_envs = DummyVectorEnv(train_envs)
        # collector
        train_collector = Collector(#Async
            policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs), exploration_noise=True)#
        )
        #test_collector = Collector(policy, test_envs)
        # log
        log_path = os.path.join(args.logdir, 'ppo') #, args.task
        #writer = SummaryWriter(log_path)
        #logger = TensorboardLogger(writer)

        def save_best_fn(policy):
            print(f"policy saved_{trader_id}")
            torch.save(policy.state_dict(), os.path.join(log_path, f"policy{trader_id}.pth"))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold

        def save_checkpoint_fn(epoch, env_step, gradient_step):
            # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            ckpt_path = os.path.join(log_path, "checkpoint.pth")
            # Example: saving by epoch num
            # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optim": optim.state_dict(),
                }, ckpt_path
            )
            return ckpt_path

        if args.resume:
            # load from existing checkpoint
            print(f"Loading agent under {log_path}")
            ckpt_path = os.path.join(log_path, "checkpoint.pth")
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=args.device)
                policy.load_state_dict(checkpoint["model"])
                optim.load_state_dict(checkpoint["optim"])
                print("Successfully restore policy and optim.")
            else:
                print("Fail to restore policy and optim.")

        # trainer
        trainer = OnpolicyTrainer(
            policy,
            train_collector,
            None,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            step_per_collect=5,
            #episode_per_collect=args.episode_per_collect,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            #logger=logger,
            resume_from_log=args.resume,
            save_checkpoint_fn=save_checkpoint_fn,
        )

        for epoch, epoch_stat, info in trainer:
            print(f"Epoch: {epoch}")
            print(epoch_stat)
            print(info)

        #assert stop_fn(info["best_reward"])
        torch.save(policy.state_dict(), os.path.join(log_path, f"policy_ppo_trader{trader_id}.pth"))
        
        if __name__ == "__main__":
            pprint.pprint(info)
            # Let's watch its performance!
            #env = gym.make(args.task)
            policy.eval()
            #collector = Collector(policy, env)
            print("start colleting result")
            result = train_collector.collect(n_step = test_steps)
            print(result)
            #print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
            
    except (KeyboardInterrupt, SystemExit):
        print("thread error catched")
        trader.disconnect()
        env.save_to_csv(f"sep_trader_{trader_id}")
        torch.save(policy.state_dict(), os.path.join(log_path, f"policy_ppo_temp{trader_id}.pth"))
      
    finally:
        env.save_to_csv(f"sep_trader_{trader_id}")

def test_ppo_resume(args=get_args()):
    args.resume = True
    test_ppo(args)


if __name__ == "__main__":
    tickers = ["CS1"]#,"CS2"
    num_proc = 7
    trader_list = []
    threads = []
    weights = [0.5, 0.5, 0.2, 0.4, 0.5, 0.6, 0.8]#
    agent_type = ["mm", "mm", "lt", "lt", "lt", "lt","lt"]#

    #torch.manual_seed(0)
    for ticker in tickers:
        if ticker == "CS1": mm_index, lt_index = 0, 0
        elif ticker == "CS2": mm_index, lt_index = 2, 5
        for i in range(num_proc):
            if agent_type[i] == "mm":
                trader_list.append(shift.Trader(f"marketmaker_rl_{mm_index+1}"))
                threads.append(Thread(target=test_ppo,args =(trader_list[i], f"mm{mm_index+1}", weights[i], agent_type[i], ticker, get_args())))        
                mm_index += 1

            elif agent_type[i] == "lt":
                trader_list.append(shift.Trader(f"liquiditytaker_rl_0{lt_index+1}"))
                threads.append(Thread(target=test_ppo,args =(trader_list[i], f"lt{lt_index+1}", weights[i], agent_type[i], ticker, get_args())))        
                lt_index += 1

    
    try:
        for i in range(len(trader_list)):#len(tickers)*
            trader_list[i].disconnect()
            trader_list[i].connect("initiator.cfg", "password")
            trader_list[i].sub_all_order_book()
            sleep(1)
            print(f"bp of {i+1}:",trader_list[i].get_portfolio_summary().get_total_bp())

        # start_time = datetime.now().replace(minute = 15, second = 0)
        # while datetime.now() < start_time:
        #     sleep(1)
        print("Starting")

        for thread in threads:  
            thread.start()
            sleep(0.01)
            
        for thread in threads:
            thread.join()
        
    except KeyboardInterrupt:
        trader_list[0].disconnect()

    finally:
        trader_list[0].disconnect()
