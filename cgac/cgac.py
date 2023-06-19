import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils_cgac import *
from model import GaussianPolicy, QNetwork, DeterministicPolicy
import gc
import ipdb
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import time
from torch.distributions import Normal

class CGAC(object):
    def __init__(self, num_inputs, action_space, args, memory_cgac, env=None):

        self.gamma = args.gamma
        self.tau = args.tau_value
        self.alpha = args.alpha
        self.args = args

        if self.args.on_policy_update:
            self.env_train = env 
            if args.env_name!='AllegroHand':
                self.env_train.clear_grad()
                self.env_train.reset()
            self.state_batch = self.env_train.obs_buf

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.lr = args.actor_lr

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.rms_obs = RunningMeanStd((num_inputs,)).to(self.device)
        self.val_running_median = 1.
        self.act_running_median = 1.
        self.state_running_median = 1.
        self.memory_cgac = memory_cgac

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.critic_hidden, args).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr, betas = args.betas)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.critic_hidden, args).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.episode_steps = np.zeros(args.batch_size)
        self.episode_rewards = torch.zeros(args.batch_size).to(self.device)
        self.total_episode_reward_hist = []
        self.episode_len_hist = []
        self.num_episodes = 0
        self.total_numsteps = 0
        self.episode_wts = torch.ones(args.batch_size, 1).to(self.device)
        self.queue_ids = set([])
        self.num_nans = 0
        self.rewards_moving_avg = 0.0
        self.targ_ent_coeff = 1.0
        self.mean_ratio = 100
        self.ep_lens = torch.zeros_like(self.env_train.progress_buf) + 10
        self.obs_dict = {}

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.log_alpha.data[0] = np.log(self.alpha)
                self.alpha_optim = Adam([self.log_alpha], lr=args.alpha_lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.actor_hidden, args, action_space, const_std=args.const_std).to(self.device)            
            self.policy_optim = Adam(self.policy.parameters(), lr=args.actor_lr, betas = args.betas)
            self.policy_avg = GaussianPolicy(num_inputs, action_space.shape[0], args.actor_hidden, args, action_space, const_std=args.const_std).to(self.device)
            hard_update(self.policy_avg, self.policy)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.actor_hidden, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.actor_lr)

    def select_action(self, state, evaluate=False):
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach()

    def compute_loss(self, out, targ):
        """
        Computes the loss for the critic network. 
        Hinge loss for TD error with hinge at 4*median.
        """
        diff = torch.abs(out - targ)
        median = diff.median(dim=0)[0].detach().clone().unsqueeze(0)
        mask = (diff < 4*median).float()
        cost = torch.square(out - targ)*(mask) + diff*8*median*(1-mask)
        cost = torch.mean(cost)
        return cost

    def update_parameters_and_collect_buffer_RL(self, memory, batch_size, updates):

        ### Learning rate schedule for the actor and critic networks ###
        if self.args.lr_schedule == 'linear':
            final_lr = self.args.final_lr
            actor_lr = max((final_lr - self.args.actor_lr) * float((updates) / self.args.max_updates) + self.args.actor_lr, final_lr)
            for param_group in self.policy_optim.param_groups:
                param_group['lr'] = actor_lr
            self.lr = actor_lr
            critic_lr = max((final_lr - self.args.critic_lr) * float((updates) / self.args.max_updates) + self.args.critic_lr, final_lr)
            for param_group in self.critic_optim.param_groups:
                param_group['lr'] = critic_lr
        elif self.args.lr_schedule == 'step':
            final_lr = self.args.final_lr
            exp = updates//self.args.lr_update_freq
            actor_lr = max(self.args.actor_lr * (self.args.lr_decay_rate**exp), final_lr)
            for param_group in self.policy_optim.param_groups:
                param_group['lr'] = actor_lr
            self.lr = actor_lr
            critic_lr = max(self.args.critic_lr * (self.args.lr_decay_rate**exp), final_lr)
            for param_group in self.critic_optim.param_groups:
                param_group['lr'] = critic_lr
        else:
            self.lr = self.args.actor_lr
        
        ### Schedule for alpha or target entropy for policy ###
        if self.args.alpha_schedule == 'linear':
            if self.automatic_entropy_tuning:
                self.targ_ent_coeff = (self.args.final_targ_ent_coeff - self.args.init_targ_ent_coeff)*(self.rewards_moving_avg - self.args.init_expected_reward)/\
                                            (self.args.peak_expected_reward - self.args.init_expected_reward) + self.args.init_targ_ent_coeff
            else:
                self.alpha = max((self.args.alpha_final - self.args.alpha_init) * float((updates) / self.args.max_updates_alpha) + self.args.alpha_init, self.args.alpha_final)
        elif self.args.alpha_schedule == 'decay':
            if updates % self.args.decay_steps == 0:
                self.alpha = max(self.args.alpha_init/(2**(updates/self.args.decay_steps)), self.args.alpha_final)

        ### Update Obs statistics and get obs ###
        self.rms_obs.update(self.env_train.obs_buf.detach().clone())
        obs_batch = self.rms_obs(self.env_train.obs_buf)

        ### Sample actions and execute it in the environment ###
        with torch.no_grad():
            action, _, _ = self.policy_avg.sample(obs_batch)
        action_batch = action
        with torch.no_grad():
            next_obs_batch, reward_batch, done_batch, info = self.env_train.step(action_batch, force_done_ids)
        next_obs_batch = self.rms_obs(next_obs_batch)
        mask_batch = (1-done_batch.unsqueeze(-1)).float()
        done_env_ids = done_batch.nonzero(as_tuple = False).squeeze(-1).cpu().numpy()

        ### Update reward statistics ###
        self.episode_steps += 1
        self.total_numsteps += self.args.batch_size
        self.episode_rewards += reward_batch
        reward_batch = reward_batch.unsqueeze(-1)
        self.reward_batch_curr = (reward_batch.clone().detach()*self.episode_wts).sum()/self.episode_wts.sum()
        self.rewards_moving_avg = 0.95*self.rewards_moving_avg + 0.05*self.reward_batch_curr
        
        ### Handling Env Terminations ###
        if len(done_env_ids) > 0:
            self.total_episode_reward_hist += self.episode_rewards[done_env_ids].cpu().numpy().tolist()
            self.episode_len_hist += self.episode_steps[done_env_ids].tolist()
            self.episode_rewards[done_env_ids] = 0
            self.episode_steps[done_env_ids] = 0
            self.num_episodes += len(done_env_ids)
            inv_mask = torch.logical_or(
                        torch.logical_or(
                            torch.isnan(info['obs_before_reset'][done_env_ids]).sum(dim=-1) > 0,
                            torch.isinf(info['obs_before_reset'][done_env_ids]).sum(dim=-1) > 0),
                        (torch.abs(info['obs_before_reset'][done_env_ids]) > 1e6).sum(dim=-1) > 0
                        )
            self.num_nans += inv_mask.float().sum()
            eplen_mask = self.env_train.progress_buf_mask[done_env_ids] == self.env_train.episode_length
            if eplen_mask.float().sum() >0:
                eplen_done_ids = done_env_ids[eplen_mask.cpu().numpy()]
                inv_mask = torch.logical_or(
                        torch.logical_or(
                            torch.isnan(info['obs_before_reset'][eplen_done_ids]).sum(dim=-1) > 0,
                            torch.isinf(info['obs_before_reset'][eplen_done_ids]).sum(dim=-1) > 0),
                        (torch.abs(info['obs_before_reset'][eplen_done_ids]) > 1e6).sum(dim=-1) > 0
                        )
                valid_mask = torch.logical_not(inv_mask)
                val_done_ids = eplen_done_ids[valid_mask.cpu().numpy()]
                if len(val_done_ids)>0:
                    mask_batch[val_done_ids] = 1.
                    next_obs_batch[val_done_ids] = self.rms_obs(info['obs_before_reset'][val_done_ids])

        ### Compute next state Q values ###
        with torch.no_grad():
            next_state_action1, next_state_log_pi, _ = self.policy_avg.sample(next_obs_batch.detach(), 1) 
            qf1_next_target1, qf2_next_target1 = self.critic_target(next_obs_batch, next_state_action1)
            min_qf_next = torch.min(qf1_next_target1, qf2_next_target1) 

        term_mask = mask_batch.clone().detach()
        
        ### Update replay buffer ###
        self.memory_cgac.add(obs_batch, action_batch, reward_batch, min_qf_next.detach().clone(), mask_batch, term_mask, next_state_log_pi.detach().clone(), self.alpha, self.episode_wts.clone().detach(), updates)


        ### Prevent suddent collapse and reset of too many envs - This can skew the distribution ###
        ### This is done by keeping a count of finished episodes and performing resets at a specific rate ###
        ### The finished episodes until they are reset are ignored for training ###
        env_thres_stop = self.env_thres = (self.args.batch_size)/(self.env_train.ep_lens[:200].float().mean()+1e-8)
        env_thres_start = max((self.args.batch_size)/(self.env_train.ep_lens.float().mean()+1e-8), 1)
        self.len_dones = len(done_env_ids)
        force_done_ids = None  
        done_env_ids_resets = list(set(done_env_ids).difference(self.queue_ids))
        if len(done_env_ids_resets) > int(env_thres_stop) + 1:
            env_thres_int = int(env_thres_stop) + 1
            self.episode_wts[done_env_ids_resets[env_thres_int:]] = 0 

            self.queue_ids.update(list(done_env_ids_resets[env_thres_int:]))
        if len(done_env_ids_resets) < int(env_thres_start):
            env_thres_int = int(env_thres_start)
            num_resets = min(env_thres_int-len(done_env_ids_resets), len(self.queue_ids)+self.args.max_rand_resets)
            num_rand_resets = max(min(num_resets - len(self.queue_ids), self.args.max_rand_resets),0)
            nids = min(env_thres_int-len(done_env_ids_resets), len(self.queue_ids))

            queue_ids = list(self.queue_ids)
            if num_rand_resets>0:
                rand_reset_ids = list(np.random.randint(self.args.batch_size, size=(num_rand_resets,)))
                reset_ids = rand_reset_ids + queue_ids
                self.memory_cgac.mask_t[:, rand_reset_ids] *= 0
            else:
                reset_ids = queue_ids[:nids]
            self.episode_wts[reset_ids] = 1
            force_done_ids = torch.tensor(reset_ids).long()

            self.queue_ids = set(queue_ids[nids:]).difference(reset_ids)
            self.env_train.reset(force_done_ids, eplenupdate=False)
            self.episode_rewards[force_done_ids] = 0
            self.episode_steps[force_done_ids] = 0
        self.len_queue = len(self.queue_ids)

        ### Update critic parameters if num_critic_updates > 1 ###
        for i in range(self.args.num_critic_updates-1):
            self.update_parameters_with_RL(updates, critic_update_only=True)
            updates += 1

        ### Update critic params, actor params and alpha ###
        return self.update_parameters_with_RL(updates)

    def update_parameters_with_RL(self, updates, critic_update_only=False):

        obs_batch_new, action_batch, reward_batch, next_q_value, mask_batch, episode_wts = self.memory_cgac.sample(self.args.batch_size_update)

        episode_wts = episode_wts/episode_wts.mean()
        qf1, qf2 = self.critic(obs_batch_new, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        
        ### CRITIC UPDATE ###

        ### Computing hinge TD errors. episode_wts are used to ignore transitions ###
        qf1_loss = self.compute_loss(qf1*episode_wts, next_q_value*episode_wts)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = self.compute_loss(qf2*episode_wts, next_q_value*episode_wts)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        qf1_loss_batch = (qf1*episode_wts - next_q_value*episode_wts)**2
        qf2_loss_batch = (qf2*episode_wts - next_q_value*episode_wts)**2
        qf_loss_batch = qf1_loss_batch + qf2_loss_batch

        qf_loss_full = qf_loss 
        self.critic_optim.zero_grad()
        qf_loss_full.backward()

        critic_grad_norm =  clip_grad_norm_(self.critic.parameters(), self.args.grad_norm)
        self.critic_optim.step()

        ### Update actor and alpha if critic_update_only is not True ###
        if not critic_update_only:
            ### ACTOR UPDATE ###
            pi, log_pi, _, x_t  = self.policy.sample(obs_batch_new.detach().clone(), with_xt=True)

            qf1_pi, qf2_pi = self.critic(obs_batch_new.detach().clone(), pi)
            min_qf_pi = torch.min(qf1_pi,qf2_pi)*episode_wts

            if self.args.clip_actor_gn:
                qpi_grad = torch.autograd.grad(min_qf_pi.mean(), pi, retain_graph=True)[0]
                ratio = (qpi_grad.abs().max(dim=0)[0]/qpi_grad.abs().median(dim=0)[0]).mean()
                self.mean_ratio = 0.95*self.mean_ratio + 0.05*ratio
                if ratio > self.mean_ratio*2:
                    qpi_grad = torch.clamp(qpi_grad, min=torch.quantile(qpi_grad, 0.5, dim=0), max=torch.quantile(qpi_grad, 0.95, dim=0)).detach().clone()

                policy_loss =  (self.alpha * log_pi*episode_wts).mean() - (qpi_grad*pi).sum()#  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            else:
                policy_loss =  ((self.alpha * log_pi*episode_wts) - min_qf_pi).mean()

            # pol_grad = torch.autograd.grad(policy_loss, pi, retain_graph=True)[0]
            # self.snr_q = (pol_grad.mean(dim=0).abs()/pol_grad.std(dim=0)).mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            actor_grad_norm = self.policy.layers[3].weight.grad.norm().item()
            self.policy_optim.step()

            ### ALPHA UPDATE ###
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi*episode_wts + self.targ_ent_coeff*self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = min(self.log_alpha.exp().detach().item(), self.alpha)
                self.log_alpha.data[0] = np.log(self.alpha)
                alpha_tlogs = self.alpha # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

            ### Update target network and policy avg network ###
            if updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.args.tau_value)
                soft_update(self.policy_avg, self.policy, self.args.tau_policy)

            return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs, log_pi.mean().item(), min_qf_pi.mean().item(), next_q_value.detach().cpu(),  \
            qf_loss_batch.detach().cpu(), actor_grad_norm, critic_grad_norm
        return None

    def update_parameters_and_collect_buffer_RL_isaac(self, memory, batch_size, updates):

        ### Learning rate schedule for the actor and critic networks ###
        if self.args.lr_schedule == 'linear':
            final_lr = self.args.final_lr
            actor_lr = max((final_lr - self.args.actor_lr) * float((updates) / self.args.max_updates) + self.args.actor_lr, final_lr)
            for param_group in self.policy_optim.param_groups:
                param_group['lr'] = actor_lr
            self.lr = actor_lr
            critic_lr = max((final_lr - self.args.critic_lr) * float((updates) / self.args.max_updates) + self.args.critic_lr, final_lr)
            for param_group in self.critic_optim.param_groups:
                param_group['lr'] = critic_lr
        elif self.args.lr_schedule == 'step':
            final_lr = self.args.final_lr
            exp = updates//self.args.lr_update_freq
            actor_lr = max(self.args.actor_lr * (self.args.lr_decay_rate**exp), final_lr)
            for param_group in self.policy_optim.param_groups:
                param_group['lr'] = actor_lr
            self.lr = actor_lr
            critic_lr = max(self.args.critic_lr * (self.args.lr_decay_rate**exp), final_lr)
            for param_group in self.critic_optim.param_groups:
                param_group['lr'] = critic_lr
        else:
            self.lr = self.args.actor_lr

        ### Schedule for alpha or target entropy for policy ###
        if self.args.alpha_schedule == 'linear':
            if self.automatic_entropy_tuning:
                self.targ_ent_coeff = max((self.args.final_targ_ent_coeff - self.args.init_targ_ent_coeff)*(self.rewards_moving_avg - self.args.init_expected_reward)/\
                                            (self.args.peak_expected_reward - self.args.init_expected_reward) + self.args.init_targ_ent_coeff, -0.4)#, self.targ_ent_coeff)#self.args.final_targ_ent_coeff)
                # if self.rewards_moving_avg > self.args.peak_expected_reward:
                #     self.alpha = min(self.alpha, 1e-5)
            else:
                self.alpha = max((self.args.alpha_final - self.args.alpha_init) * float((updates) / self.args.max_updates_alpha) + self.args.alpha_init, self.args.alpha_final)
        elif self.args.alpha_schedule == 'decay':
            if updates % self.args.decay_steps == 0:
                self.alpha = max(self.args.alpha_init/(2**(updates/self.args.decay_steps)), self.args.alpha_final)

        ### Update Obs statistics and get obs ###
        obs_dict, dones = self._env_reset_done()
        obs_buf = obs_dict['obs']
        self.rms_obs.update(obs_buf)
        obs_batch = self.rms_obs(obs_buf)

        ### Sample actions and execute it in the environment ###
        with torch.no_grad():
            action, _, _ = self.policy_avg.sample(obs_batch)
        action_batch = action
        next_obs_batch, reward_batch, done_batch, info = self.env_train.step(action_batch)
        next_obs_batch = next_obs_batch['obs']
        next_obs_batch = self.rms_obs(next_obs_batch)
        mask_batch = (1-done_batch.unsqueeze(-1)).float()
        done_env_ids = done_batch.nonzero(as_tuple = False).squeeze(-1).cpu().numpy()
        term_mask = mask_batch.clone()

        ### Update reward statistics ###
        self.episode_steps += 1
        self.total_numsteps += self.args.batch_size
        self.episode_rewards += reward_batch.detach().clone()
        reward_batch = reward_batch.unsqueeze(-1)
        self.reward_batch_curr = (reward_batch.clone().detach()*self.episode_wts).sum()/self.episode_wts.sum()
        self.rewards_moving_avg = 0.95*self.rewards_moving_avg + 0.05*self.reward_batch_curr
        
        ### Handling Env Terminations
        if len(done_env_ids) > 0:
            self.total_episode_reward_hist += self.episode_rewards[done_env_ids].cpu().numpy().tolist()
            self.episode_len_hist += self.episode_steps[done_env_ids].tolist()
            self.episode_rewards[done_env_ids] = 0
            self.episode_steps[done_env_ids] = 0
            self.num_episodes += len(done_env_ids)
            inv_mask = torch.logical_or(
                        torch.logical_or(
                            torch.isnan(next_obs_batch[done_env_ids]).sum(dim=-1) > 0,
                            torch.isinf(next_obs_batch[done_env_ids]).sum(dim=-1) > 0),
                        (torch.abs(next_obs_batch[done_env_ids]) > 1e6).sum(dim=-1) > 0
                        )
            self.num_nans += inv_mask.float().sum()
            self.ep_lens = torch.cat([self.env_train.progress_buf[done_env_ids].clone(), self.ep_lens], dim=0)[:200]
            eplen_mask = self.env_train.progress_buf[done_env_ids] == self.env_train.max_episode_length  # Need to check
            if eplen_mask.float().sum() >0:
                eplen_done_ids = done_env_ids[eplen_mask.cpu().numpy()]
                inv_mask = torch.logical_or(
                        torch.logical_or(
                            torch.isnan(next_obs_batch[eplen_done_ids]).sum(dim=-1) > 0,
                            torch.isinf(next_obs_batch[eplen_done_ids]).sum(dim=-1) > 0),
                        (torch.abs(next_obs_batch[eplen_done_ids]) > 1e6).sum(dim=-1) > 0
                        )
                valid_mask = torch.logical_not(inv_mask)
                val_done_ids = eplen_done_ids[valid_mask.cpu().numpy()]
                if len(val_done_ids)>0:
                    mask_batch[val_done_ids] = 1.

        ### Compute next state Q values ###
        with torch.no_grad():
            next_state_action1, next_state_log_pi, _ = self.policy_avg.sample(next_obs_batch.detach(), 1) 
            qf1_next_target1, qf2_next_target1 = self.critic_target(next_obs_batch, next_state_action1)
            min_qf_next = torch.min(qf1_next_target1, qf2_next_target1)

        ### Update replay buffer ###
        self.memory_cgac.add(obs_batch, action_batch, reward_batch, min_qf_next.detach().clone(), mask_batch, term_mask, next_state_log_pi.detach().clone(), self.alpha, self.episode_wts.clone().detach(), updates)

        ### Prevent suddent collapse and reset of too many envs - This can skew the distribution ###
        ### This is done by keeping a count of finished episodes and performing resets at a specific rate ###
        ### The finished episodes until they are reset are ignored for training ###
        env_thres_stop = self.env_thres = (self.args.batch_size)/(self.ep_lens[:200].float().mean()+1e-8)
        env_thres_start = (self.args.batch_size)/(self.ep_lens.float().mean()+1e-8)
        self.len_dones = len(done_env_ids)
        done_env_ids_resets = list(set(done_env_ids).difference(self.queue_ids))
        force_done_ids = None
        if len(done_env_ids_resets) > int(env_thres_stop) + 1:
            env_thres_int = int(env_thres_stop) + 1
            self.episode_wts[done_env_ids_resets[env_thres_int:]] = 0 
            self.queue_ids.update(list(done_env_ids_resets[env_thres_int:]))
        if len(done_env_ids_resets) < int(env_thres_start):
            env_thres_int = int(env_thres_start)
            num_resets = min(env_thres_int-len(done_env_ids_resets), len(self.queue_ids)+self.args.max_rand_resets)
            num_rand_resets = max(min(num_resets - len(self.queue_ids), self.args.max_rand_resets),0)
            nids = min(env_thres_int-len(done_env_ids_resets), len(self.queue_ids))
            queue_ids = list(self.queue_ids)
            if num_rand_resets>0:
                rand_reset_ids = list(np.random.randint(self.args.batch_size, size=(num_rand_resets,)))
                reset_ids = rand_reset_ids + queue_ids
                self.memory_cgac.mask_t[:, rand_reset_ids] *= 0
            else:
                reset_ids = queue_ids[:nids]
            if len(reset_ids) > 0:
                self.episode_wts[reset_ids] = 1
                force_done_ids = torch.tensor(reset_ids).long()
                self.queue_ids = set(queue_ids[nids:]).difference(reset_ids)
                self.env_train.reset_idx(force_done_ids, force_done_ids)
                self.episode_rewards[force_done_ids] = 0
                self.episode_steps[force_done_ids] = 0
        self.len_queue = len(self.queue_ids)

        ### Update critic parameters if num_critic_updates > 1 ###
        for i in range(self.args.num_critic_updates-1):
            self.update_parameters_with_RL_isaac(updates, critic_update_only=True)
            updates += 1

        ### Update critic params, actor params and alpha ###
        return self.update_parameters_with_RL_isaac(updates)

    def update_parameters_with_RL_isaac(self, updates, critic_update_only=False):
        
        obs_batch_new, action_batch, reward_batch, next_q_value, mask_batch, episode_wts = self.memory_cgac.sample(self.args.batch_size_update)

        episode_wts = episode_wts/episode_wts.mean()
        qf1, qf2 = self.critic(obs_batch_new, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        
        ### CRITIC UPDATE ###

        ### Computing hinge TD errors. episode_wts are used to ignore transitions ###
        qf1_loss = self.compute_loss(qf1*episode_wts, next_q_value*episode_wts)
        qf2_loss = self.compute_loss(qf2*episode_wts, next_q_value*episode_wts)
        qf_loss = qf1_loss + qf2_loss

        qf1_loss_batch = (qf1*episode_wts - next_q_value*episode_wts)**2
        qf2_loss_batch = (qf2*episode_wts - next_q_value*episode_wts)**2
        qf_loss_batch = qf1_loss_batch + qf2_loss_batch

        qf_loss_full = qf_loss 
        self.critic_optim.zero_grad()
        qf_loss_full.backward()

        critic_grad_norm =  clip_grad_norm_(self.critic.parameters(), self.args.grad_norm)
        self.critic_optim.step()

        ### Update actor and alpha if critic_update_only is not True ###
        if not critic_update_only:
            ### ACTOR UPDATE ###
            pi, log_pi, _  = self.policy.sample(obs_batch_new.detach().clone())

            qf1_pi, qf2_pi = self.critic(obs_batch_new.detach().clone(), pi)
            min_qf_pi = torch.min(qf1_pi,qf2_pi)*episode_wts

            if self.args.clip_actor_gn:
                qpi_grad = torch.autograd.grad(min_qf_pi.mean(), pi, retain_graph=True)[0]
                ratio = (qpi_grad.abs().max(dim=0)[0]/qpi_grad.abs().median(dim=0)[0]).mean()
                self.mean_ratio = 0.95*self.mean_ratio + 0.05*ratio
                if ratio > self.mean_ratio*2:
                    qpi_grad = torch.clamp(qpi_grad, min=torch.quantile(qpi_grad, 0.5, dim=0), max=torch.quantile(qpi_grad, 0.95, dim=0)).detach().clone()

                policy_loss =  (self.alpha * log_pi*episode_wts).mean() - (qpi_grad*pi).sum()#  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            else:
                policy_loss =  ((self.alpha * log_pi*episode_wts) - min_qf_pi).mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            actor_grad_norm = self.policy.layers[3].weight.grad.norm().item()
            self.policy_optim.step()

            ### ALPHA UPDATE ###
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi*episode_wts + self.targ_ent_coeff*self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = min(self.log_alpha.exp().detach().item(), self.alpha)
                self.log_alpha.data[0] = np.log(self.alpha)
                alpha_tlogs = self.alpha # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

            ### Update target network and policy avg network ###
            if updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.args.tau_value)
                soft_update(self.policy_avg, self.policy, self.args.tau_policy)

            return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs, log_pi.mean().item(), min_qf_pi.mean().item(), next_q_value.detach().cpu(),  \
            qf_loss_batch.detach().cpu(), actor_grad_norm, critic_grad_norm
        return None


    def _env_reset_done(self):
        """Reset the environment.
        Returns:
            Observation dictionary, indices of environments being reset
        """
        done_env_ids = self.env_train.reset_buf.nonzero(as_tuple=False).flatten()
        goal_env_ids = self.env_train.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(done_env_ids) == 0:
            self.env_train.reset_target_pose(goal_env_ids, apply_reset=True)

        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.env_train.reset_target_pose(goal_env_ids)

        if len(done_env_ids) > 0:
            self.env_train.reset_idx(done_env_ids, goal_env_ids)

        if len(goal_env_ids) > 0 or len(done_env_ids) > 0:
            self.env_train.compute_observations()

        self.obs_dict["obs"] = torch.clamp(self.env_train.obs_buf, -self.env_train.clip_obs, self.env_train.clip_obs).to(self.env_train.rl_device)

        return self.obs_dict, done_env_ids

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/cgac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

