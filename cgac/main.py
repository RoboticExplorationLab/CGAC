
import sys, os
import argparse
import datetime
import time
import numpy as np
import itertools
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="SNUHumanoidEnv", choices=["AntEnv", "HumanoidEnv", "SNUHumanoidEnv", "CartPoleSwingUpEnv", "CheetahEnv", "HopperEnv", "AllegroHand"])
parser.add_argument('--policy', default="Gaussian", help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau_value', type=float, default=0.005, metavar='G', help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--tau_policy', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--final_lr', type=float, default=1e-4)
parser.add_argument('--alpha', type=float, default=0.1, metavar='G', help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--alpha_final', type=float, default=2e-5)
parser.add_argument('--no_automatic_entropy_tuning', action="store_true", help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size_update', type=int, default=8192*2)
parser.add_argument('--num_actors', type=int, default=4096)
parser.add_argument('--num_steps', type=int, default=500000001)
parser.add_argument('--critic_hidden', nargs='+', type=int, default=[512, 512, 256])
parser.add_argument('--actor_hidden', nargs='+', type=int, default=[512, 256])
parser.add_argument('--critic_act', type=str, default='elu', choices=['elu', 'tanh', 'relu'])
parser.add_argument('--actor_act', type=str, default='elu', choices=['elu', 'tanh', 'relu'])
parser.add_argument('--num_critic_updates', type=int, default=2)
parser.add_argument('--val_horizon', type=int, default=32)
parser.add_argument('--num_steps_buffer', type=int, default=32)
parser.add_argument('--betas', nargs='+',  type=float, default=[0.9, 0.999])
parser.add_argument('--lam', type=float, default=0.95)
parser.add_argument('--no_const_std', action='store_true')
parser.add_argument('--grad_norm', type=float, default=20)
parser.add_argument('--actor_grad_norm', type=float, default=4)
parser.add_argument('--clip_actor_gn', action='store_true')
parser.add_argument('--max_updates', type=int, default=20000)
parser.add_argument('--lr_schedule', type=str, default='linear', choices=['linear', 'decay', 'constant'])
parser.add_argument('--alpha_schedule', type=str, default='linear', choices=['linear', 'decay', 'constant'])
parser.add_argument('--final_targ_ent_coeff', type=float, default=3.5)
parser.add_argument('--init_targ_ent_coeff', type=float, default=0.2)
parser.add_argument('--peak_expected_reward', type=float, default=7.5)
parser.add_argument('--init_expected_reward', type=float, default=1.5)
parser.add_argument('--critic_method', type=str, default='gae-return', choices=['gae-return', 'td-lambda', 'one-step'])
parser.add_argument('--episode_length', type=int, default=1000)
parser.add_argument('--no_stochastic_init', action='store_true')
parser.add_argument('--policy_clip', action='store_true')
parser.add_argument('--cuda', action="store_true")
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N', help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--id', type=str, default='0')
parser.add_argument('--desc', type=str, default='')
parser.add_argument('--final', action='store_true')
parser.add_argument('--start_steps', type=int, default=80000, metavar='N', help='Steps sampling random actions (default: 10000)')
parser.add_argument('--replay_size', type=int, default=1000000)
parser.add_argument('--test', action="store_true")
parser.add_argument('--eval', type=bool, default=True, help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--on_policy_update', action='store_true')
parser.add_argument('--reduction_rate_updates', type=int, default=8000)
# parser.add_argument('--num_steps_buffer', type=int, default=32)
parser.add_argument('--max_updates_alpha', type=int, default=8000)
parser.add_argument('--decay_steps', type=int, default=2000)
parser.add_argument('--lr_update_freq', type=int, default=1000)
parser.add_argument('--lr_decay_rate', type=float, default=0.75)
args = parser.parse_args()


args.automatic_entropy_tuning = not args.no_automatic_entropy_tuning
args.batch_size = args.num_actors
args.critic_lr = args.lr
args.actor_lr = args.lr
args.alpha_lr = args.lr*10
args.alpha_init = args.alpha*4
# args.alpha_final = args.alpha/4
args.max_rand_resets = args.batch_size//args.episode_length
args.horizon_shrink_steps = 12000/args.num_steps_buffer
args.const_std = not args.no_const_std
args.no_grad_train = not args.grad_train
args.stochastic_init = not args.no_stochastic_init
device_str = "cuda:0" if args.cuda else "cpu"
device = args.device = device_str 


if args.env_name=='AllegroHand':
    rl_games_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../rl_games/'))
    sys.path.append(rl_games_dir)
    import isaacgym
    import isaacgymenvs
    import torch
    env = isaacgymenvs.make(
            args.seed, 
            args.env_name, 
            args.batch_size, 
            device_str,
            device_str,
            graphics_device_id=0,
            headless=True)
    env.actions = torch.zeros((env.num_envs, env.num_actions), device=device, dtype=torch.float)
    args.episode_length = 600
    args.max_rand_resets = 0 
else:
    import dflex as df
    import envs
    import torch
    from utils.common import *

    if args.env_name == "HumanoidEnv":
        args.MM_caching_frequency = 48
    elif args.env_name == "SNUHumanoidEnv":
        args.MM_caching_frequency = 8
    elif args.env_name == "AntEnv":
        args.MM_caching_frequency = 16

    seeding(args.seed)
    env_fn = getattr(envs, args.env_name)
    env = env_fn(num_envs = args.batch_size, \
                    device = args.device, \
                    render = False, \
                    seed = args.seed, \
                    episode_length=args.episode_length, \
                    stochastic_init = args.stochastic_init, \
                    MM_caching_frequency = args.MM_caching_frequency, \
                    no_grad=args.no_grad_train)

from cgac.cgac import CGAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import VectorizedReplayBufferIsaacSAC
from utils_cgac import *
from utils.common import *

seeding(args.seed)
memory_cgac = VectorizedReplayBufferIsaacSAC(env.obs_space.shape, env.act_space.shape, args.batch_size, args.num_steps_buffer, args.device, gamma=args.gamma, lam=args.lam, horizon=args.val_horizon, critic_method=args.critic_method)
agent = CGAC(env.obs_space.shape[0], env.act_space, args, memory_cgac, env)

if not args.test:
    save_dir = f'runs/hp/{args.env_name}/RL_final/tauval{args.tau_value}pi{args.tau_policy}_{args.actor_act}{args.actor_hidden}_\
{args.critic_act}{args.critic_hidden}_actors{args.num_actors}bszup{args.batch_size_update}_alphaautolin{int(args.final_targ_ent_coeff)}_\
{args.alpha}fin{args.alpha_final}_criticups{args.num_critic_updates}_bufs{args.num_steps_buffer}h{args.val_horizon}_seed{args.seed}_\
piclip{args.policy_clip}_{args.desc}'
    print('save_dir : ', save_dir)
    writer = SummaryWriter(save_dir) 

if args.env_name=='AllegroHand':
    RL_update_func = agent.update_parameters_and_collect_buffer_RL_isaac
else:
    RL_update_func = agent.update_parameters_and_collect_buffer_RL

# Training Loop
total_numsteps = 0
updates = 0
total_updates = 0
num_episodes = 0
episode_steps = np.zeros(args.num_actors)
episode_rewards = torch.zeros(args.num_actors).to(device)
total_episode_reward_hist = []
episode_len_hist = []
agent.times = []
last = time.time()
start_time = time.time()

for i_episode in itertools.count(1):
    critic_1_grad_state_loss = None
    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, log_pi, min_qf_pi, next_values, \
    qf_loss_batch, actor_grad_norm, critic_grad_norm, = RL_update_func(memory_cgac, args.batch_size, updates)
    if not args.test and not args.final:
        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
        writer.add_scalar('loss/policy', policy_loss, updates)
        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
        writer.add_scalar('loss/log_pi', log_pi, updates)
        writer.add_scalar('loss/min_qf_pi', min_qf_pi, updates)
        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
        writer.add_scalar('entropy_temprature/actor_grad_norm', actor_grad_norm, updates)
        writer.add_scalar('entropy_temprature/critic_grad_norm', critic_grad_norm, updates)
        writer.add_scalar('losses/val_targ_max', next_values.max().item(), updates)
        writer.add_scalar('losses/val_targ_mean', next_values.mean().item(), updates)
        writer.add_scalar('losses/val_targ_min', next_values.min().item(), updates)

        writer.add_scalar('losses/loss_qf_max', qf_loss_batch.max().item(), updates)
        writer.add_scalar('losses/loss_qf_mean', qf_loss_batch.mean().item(), updates)
        writer.add_scalar('losses/loss_qf_min', qf_loss_batch.min().item(), updates)
        writer.add_scalar('losses/loss_qf_median', qf_loss_batch.median().item(), updates)
        writer.add_scalar('entropy_temprature/num_nans', agent.num_nans, updates)

        writer.add_scalar('agent_rewards/rew1', agent.episode_rewards[0], i_episode)
        writer.add_scalar('agent_rewards/rew10', agent.episode_rewards[10], i_episode)
        writer.add_scalar('agent_rewards/rew100', agent.episode_rewards[20], i_episode)
        writer.add_scalar('agent_rewards/rew1k', agent.episode_rewards[40], i_episode)
        writer.add_scalar('agent_rewards/eplenavg', agent.env_train.progress_buf.clone().float().mean().item(), i_episode)
        writer.add_scalar('agent_done_stats/done_len', agent.len_dones, i_episode)
        writer.add_scalar('agent_done_stats/queue_len', agent.len_queue, i_episode)
        writer.add_scalar('agent_done_stats/env_thres', agent.env_thres, i_episode)
        writer.add_scalar('agent_done_stats/memory_cgac_horizon', memory_cgac.h, i_episode)
        writer.add_scalar('agent_done_stats/sum_episode_wts', (1-agent.episode_wts).sum(), i_episode)
        writer.add_scalar('agent_done_stats/agent_curr_rew', agent.reward_batch_curr, i_episode)

    updates += 1
    total_updates += args.num_critic_updates

    total_numsteps = updates*args.batch_size
    if total_numsteps > args.num_steps:
        break

    if i_episode%100 == 0:
        time_taken = time.time() - last
        last = time.time()
        total_time = time.time() - start_time
        if len(agent.total_episode_reward_hist)>0:
            num_test_avg = 100
            if not args.test:
                writer.add_scalar('reward/train', np.array(agent.total_episode_reward_hist[-num_test_avg:]).mean(), i_episode)
                writer.add_scalar('reward/train_recent', agent.total_episode_reward_hist[-1], i_episode)
                writer.add_scalar('reward/ep_len', np.array(agent.episode_len_hist[-num_test_avg:]).mean(), i_episode)
                writer.add_scalar('reward/rew_eplen', np.array(agent.total_episode_reward_hist[-num_test_avg:]).mean()/np.array(agent.episode_len_hist[-num_test_avg:]).mean(), i_episode)
                writer.add_scalar('reward/train_agent', np.array(agent.total_episode_reward_hist[-num_test_avg:]).mean(), i_episode)
                writer.add_scalar('reward/ep_len_agent', np.array(agent.episode_len_hist[-num_test_avg:]).mean(), i_episode)         
                writer.add_scalar('reward/rew_eplen_agent', np.array(agent.total_episode_reward_hist[-num_test_avg:]).mean()/np.array(agent.episode_len_hist[-num_test_avg:]).mean(), i_episode)
                writer.add_scalar('reward/train_time', np.array(agent.total_episode_reward_hist[-num_test_avg:]).mean(), total_time)
                writer.add_scalar('reward/ep_len_time', np.array(agent.episode_len_hist[-num_test_avg:]).mean(), total_time)
                writer.add_scalar('reward/train_steps', np.array(agent.total_episode_reward_hist[-num_test_avg:]).mean(), total_numsteps)
                writer.add_scalar('reward/ep_len_steps', np.array(agent.episode_len_hist[-num_test_avg:]).mean(), total_numsteps)
                # writer.add_scalar('variance/snr_q', agent.snr_q, total_numsteps)
                # writer.add_scalar('variance/snr_q_iter', agent.snr_q, i_episode)
                # writer.add_scalar('variance/ppo_snr', agent.ppo_snr, total_numsteps)            
                # writer.add_scalar('variance/ppo_snr_iter', agent.ppo_snr, i_episode)
                # writer.add_scalar('variance/ppo_snr_adv', agent.ppo_snr_adv, total_numsteps)    
                # writer.add_scalar('variance/ppo_snr_adv_iter', agent.ppo_snr_adv, i_episode)
            print(f"iters: {i_episode}, total numsteps: {total_numsteps}, num ep: {agent.num_episodes}, episode steps: {np.array(agent.episode_len_hist[-num_test_avg:]).mean()}, reward: {np.array(agent.total_episode_reward_hist[-num_test_avg:]).mean()}, progress_buf: {env.progress_buf.min()}, time: {time_taken}, lr: {agent.lr}, num_nans: {agent.num_nans}")
        else:
            print(f"iters: {i_episode}, total numsteps: {total_numsteps}, num ep: {agent.num_episodes}, progress_buf: {env.progress_buf.min()}, time: {time_taken}, lr: {agent.lr}, num_nans: {agent.num_nans}")
        print(agent.alpha)

env.close()

