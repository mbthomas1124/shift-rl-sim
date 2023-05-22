import shift
from time import sleep
from multiprocessing import Process, Queue
    
class SHIFT_ENV_BASE:
    '''
    Launch traders using multi processing
    '''
    
    launched_traders = []
    
    processes = []
    
    available_traders = tuple((f"test00{i}" for i in range(1, 10)))
    
    config = {"initiator_path": "./initiator.cfg", 
              }
    
    def __init__(self, 
                 username, 
                 password, 
                 trade_sym,
                 message_listen_freq = 50):
        self.trader_cfg = {'username': username,
                           'password': password,
                           'symbol': trade_sym, 
                           'message_listen_freq': message_listen_freq
                #    'step_time': 2
                          }
        
        self.message_q = Queue()
        self.collect_q = Queue()
        
        self.p = Process(target = self._main, args = (username, 
                                                 password, 
                                                 trade_sym, 
                                                 self.message_q, 
                                                 self.collect_q))
        
        self.p.start()
        
        self.processes.append(self.p)
        
        
    def _main(self, username, password, trade_sym, message_q, collect_q):
        self._init_trader(username, password, trade_sym)
        
        while True:
            m = message_q.get()
            if m[0] == 0:
                ret = self._step(*m[1])
            elif m[0] == 1:
                ret = self._reset(*m[1])
            elif m[0] == 9: # kill process
                self._del()
                break
            collect_q.put(ret)
                
        
        
    def _step(self, action):
        self._print("step function called")
        time.sleep(2)
        self._print("step function ended")
        return self.trader_cfg['username'], 0
        
        
    def _reset(self):
        self._print("reset function called")
        return self.trader_cfg['username'], 0
        
        
    def _print(self, info):
        print(f"Agent {self.trader_cfg['username']}: "+info)
        

    def _init_trader(self, username, password, trade_sym):
        assert username in self.available_traders, f'Trader {username} is not available.'
        assert username not in self.launched_traders, f'Trader {username} has been created.'
        
        self.trader_cfg = {'username': username,
                           'password': password,
                           'symbol': trade_sym, 
                        #    'step_time': 2
                           }
        
        self.trader = shift.Trader(username)
        
        if not self.trader.connect(self.config["initiator_path"], password):
            raise RuntimeError(f"Connection for trader {username} failed.")

        if not self.trader.sub_order_book(trade_sym):
            raise RuntimeError(f"LOB subscription for trader {username} failed.")
        
        self.launched_traders.append(username)
        
    def _del(self):
        self.trader.disconnect()
        
    def send_signal(self, signal, message = ()):
        '''
        0 -- step function
        1 -- reset function
        9 -- kill the service
        '''
        self.message_q.put((signal, message))
        
    def collect(self):
        return self.collect_q.get()
        
    
    def step(self, action):
        self.send_signal(0, (action, ))
        return self.collect()
        
    def reset(self):
        self.send_signal(1)
        return self.collect()

 
    @classmethod
    def make(cls,
             username, 
             symbol, 
             password = "password"):
    
        env = cls(username, password, symbol)
        return env
    
    def __del__(self):
        try:
            print(f"Disconnecting {self.trader_cfg['username']}")
            self.p.kill()
        except AttributeError:
            pass
        
        
class SHIFT_MA_ENV:
    def __init__(self, trader_list) -> None:
        self.agents = []
        for trader in trader_list:
            self.agents.append(SHIFT_ENV_BASE.make(trader, "AAPL"))
    
    def step(self, actions):
        # distribute actions
        for i, a in enumerate(actions):
            self.agents[i].send_signal(0, (a, ))
            
        # collect results (TODO: need optimization)
        results = []
        for agent in self.agents:
            results.append(agent.collect())
            
        return results
    
    def reset(self):
        for agent in self.agents:
            self.agents[i].send_signal(1)
            
        # collect results (TODO: need optimization)
        results = []
        for agent in self.agents:
            results.append(agent.collect())
    
"""if __name__ == "__main__":
    # env1 = SHIFT_ENV_BASE.make('test001', 'AAPL')
    # env2 = SHIFT_ENV_BASE.make('test002', 'IBM')
    
    # env1.step(1)
    # env2.step(1)
    traders = list((f"test00{i}" for i in range(1, 6)))
    env = SHIFT_MA_ENV(traders)
    
    for i in range(20):
        time.sleep(5)
        returns = env.step([1,2,3,4,5])
        print(returns)"""



import argparse
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic

from RL_env_mm_split_step import SHIFT_env as Env



class Shared_env():
    def __init__(self, num_env, trader_list, sleep_t, info_step, nTimeStep, ODBK_range, symbol, max_order_list_length, weights=None, tar_ms = None) -> None:
        self.num = num_env
        self.sleep = sleep_t
        self.envs = []
        for i in range(num_env):
            self.envs.append(Env(trader = trader_list[i],
                                        rl_t = sleep_t,
                                        info_step= info_step,
                                        nTimeStep=nTimeStep,
                                        ODBK_range=ODBK_range,
                                        symbol=symbol,
                                        max_order_list_length=max_order_list_length,
                                        weight = weights[i]#,tar_m = tar_ms[i]
                                        ))
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def step(self, action):
        # print("action",action)
        states = []###########numpy array with preset space
        rewards = []
        dones = []
        infos = []
        for i in range(self.num):
            self.envs[i].step1(action[i])
        
        sleep(self.sleep)

        for i in range(self.num):
            state, reward, done, info = self.envs[i].step2(action[i])
            states.append((state))
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        rewards = np.array(rewards)

        
        print("reward", rewards)
        print("states", states)
        return states, rewards, False, {}

    def reset(self):
        states = []
        for env in self.envs:
            state = env.reset()
            states.append(state)
        return states
    
    def save_to_csv(self):
        for i in range(self.num):
            self.envs[i].save_to_csv(f"new_mm_{i}")
        
    def __call__(self, *args, **kwds):
        return self

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--task', type=str, default='Pendulum-v1')
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--episode-per-collect', type=int, default=20)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    #parser.add_argument('--step-per-collect', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
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


def test_ppo(args=get_args(), trader_list=[]):
    step_per_collect = 10
    test_steps = 300
    try:
        env = Shared_env(len(trader_list), trader_list = trader_list,
                sleep_t = 3,
                info_step= 0.25,
                nTimeStep=10,
                ODBK_range=5,
                symbol='CS1',
                max_order_list_length=10,
                weights = [1,0.5,0]#,tar_ms = [0.3,0.7]
                )  

        args.state_shape = env.observation_space.shape  # (6,)
        args.action_shape = env.action_space.shape      # (2,)
        args.max_action = env.action_space.high[0]#1
        # if args.reward_threshold is None:
        #     default_reward_threshold = {"Pendulum-v0": -250, "Pendulum-v1": -250}
        #     args.reward_threshold = default_reward_threshold.get(
        #         args.task, env.spec.reward_threshold
        #     )
        # you can also use tianshou.env.SubprocVectorEnv
        # train_envs = gym.make(args.task)
        train_envs = DummyVectorEnv([env]#SubprocVectorEnv(#DummyVectorEnv([env]#
            #[lambda: gym.make(args.task) for _ in range(args.training_num)]
        )
        # test_envs = gym.make(args.task)
        test_envs = DummyVectorEnv([env]#SubprocVectorEnv(#
            #[lambda: gym.make(args.task) for _ in range(args.test_num)]
        )
        # seed
        # np.random.seed(args.seed)
        # torch.manual_seed(args.seed)
        # train_envs.seed(args.seed)
        # test_envs.seed(args.seed)
        # model
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
            return Independent(Normal(*logits), 1)

        policy = PPOPolicy(
            actor,
            critic,
            optim,
            dist,
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
        # collector
        train_collector = Collector(
            policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
        )
        #test_collector = Collector(policy, test_envs)
        log_path = os.path.join(args.logdir, 'dqn') #, args.task
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)

        def save_best_fn(policy):
            print(f"policy saved")
            torch.save(policy.state_dict(), os.path.join(log_path, f"policy.pth"))

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
            step_per_collect=step_per_collect,
            #episode_per_collect=args.episode_per_collect,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            resume_from_log=args.resume,
            save_checkpoint_fn=save_checkpoint_fn,
        )

        for epoch, epoch_stat, info in trainer:
            print(f"Epoch: {epoch}")
            print(epoch_stat)
            print(info)
        #assert stop_fn(info["best_reward"])
        torch.save(policy.state_dict(), os.path.join(log_path, f"policy_ppo6Hours.pth"))

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
    
    except KeyboardInterrupt:
        print("thread error catched")
        env.save_to_csv()
        for i in range(num_mm):
            trader_list[i].disconnect()
        #torch.save(policy.state_dict(), os.path.join(log_path, f"policy_ppo_temp{trader_id}.pth"))
      
    finally:
        env.save_to_csv()
        for i in range(num_mm):
            trader_list[i].disconnect()

# def test_ppo_resume(args=get_args()):
#     args.resume = True
#     test_ppo(args)


if __name__ == "__main__":
    num_mm = 3
    trader_list = []
    for i in range(num_mm):
        trader_list.append(shift.Trader(f"marketmaker_rl_{i+1}"))

    for i in range(len(trader_list)):
        trader_list[i].disconnect()
    for i in range(num_mm):
            print(i)
            trader_list[i].connect("initiator.cfg", "password")
            trader_list[i].sub_all_order_book()
            sleep(1)
    for i in range(num_mm):
            print(f"bp of {i+1}:",trader_list[i].get_portfolio_summary().get_total_bp())

    try:
        
        # start_time = datetime.now().replace(minute = 15, second = 0)
        # while datetime.now() < start_time:
        #     sleep(1)
        print("Starting")
        test_ppo(trader_list=trader_list)

    except KeyboardInterrupt:
        for i in range(num_mm):
            trader_list[i].disconnect()
