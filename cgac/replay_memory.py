import random
import numpy as np
import torch 
import ipdb


class VectorizedReplayBufferIsaacSAC:
    def __init__(self, obs_shape, action_shape, batch_size, num_steps, device, gamma=0.99, lam=0.95, horizon=200000, critic_method='td-lambda'):# 
        """Create Vectorized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        See Also
        --------
        ReplayBuffer.__init__
        """
        self.device = device
        self.out_device = device
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.critic_method = critic_method
        capacity = batch_size*num_steps
        self.lam = lam
        self.gamma = gamma
        self.h = min(horizon, num_steps)
        self.minhorizon = 1


        self.actions = torch.empty((num_steps, batch_size, *action_shape), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        self.masks = torch.empty((num_steps, batch_size, 1), dtype=torch.float32, device=self.device) 
        self.term_masks = torch.empty((num_steps, batch_size, 1), dtype=torch.float32, device=self.device) 
        self.next_values = torch.empty((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        self.next_state_log_pi = torch.empty((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        self.states = torch.empty((num_steps, batch_size, *obs_shape), dtype=torch.float32, device=self.device)
        self.episode_wts = torch.empty((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        self.val_cur = torch.empty((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        
        self.gamma_k = torch.ones((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        self.sigmar = torch.zeros((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        self.Vt_new = torch.zeros((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        self.lam_t = torch.ones((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        self.Vt_out = torch.zeros((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        self.mask_t = torch.ones((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        self.sigmaG = torch.zeros((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        self.G_prev = torch.zeros((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        self.gamma_lam_k = torch.zeros((num_steps, batch_size, 1), dtype=torch.float32, device=self.device)
        self.sample_start_idxs = torch.zeros((batch_size,), dtype=torch.float32, device=self.device)
        self.num_episodes_passed = torch.zeros((batch_size,), dtype=torch.float32, device=self.device)


        self.capacity = capacity
        self.idx = 0
        self.full = False
        
    @torch.no_grad()
    def add(self, states, actions, rewards, next_q_value, masks, term_masks, next_state_log_pi, alpha, episode_wts, updates):
        num_observations = self.batch_size
        if self.idx >= self.num_steps:
            self.full = True
            self.idx = 0
        self.actions[self.idx, :] = actions
        self.states[self.idx, :] = states
        self.rewards[self.idx, :] = rewards
        self.masks[self.idx, :] = masks
        self.term_masks[self.idx, :] = term_masks
        self.next_values[self.idx, :] = next_q_value
        self.next_state_log_pi[self.idx, :] = next_state_log_pi
        self.episode_wts[self.idx, :] = episode_wts
        self.alpha = alpha
        self.compute_target_values()
        self.idx += 1

    @torch.no_grad()
    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obses: torch tensor
            batch of observations
        actions: torch tensor
            batch of actions executed given obs
        rewards: torch tensor
            rewards received as results of executing act_batch
        next_obses: torch tensor
            next set of observations seen after executing act_batch
        not_dones: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not
        not_dones_no_max: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not, specifically exlcuding maximum episode steps
        """

        idxs_steps = torch.randint(0,
                            self.num_steps if self.full else self.idx, 
                            (batch_size,), device=self.device)
        idxs_bsz = torch.randint(0,
                            self.batch_size, 
                            (batch_size,), device=self.device)
        states = self.states[idxs_steps, idxs_bsz]
        actions = self.actions[idxs_steps, idxs_bsz]
        rewards = self.rewards[idxs_steps, idxs_bsz]
        masks = self.masks[idxs_steps, idxs_bsz]
        target_val = self.Vt_out[idxs_steps, idxs_bsz]
        episode_wts = self.episode_wts[idxs_steps, idxs_bsz]

        return states, actions, rewards, target_val, masks, episode_wts

    def __len__(self):
        return self.idx

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == 'one-step':
            self.Vt_out[self.idx] = self.rewards[self.idx] + self.gamma * self.next_values[self.idx] * self.masks[self.idx]
        elif self.critic_method == 'td-lambda':
            if self.full:
                if self.idx>=self.h-1:
                    start_idx = self.idx - self.h + 1
                else:
                    start_idx = 0
                    end_idx = self.idx - self.h + 1
                    mask = self.masks[self.idx]
                    self.sigmar[end_idx:] = self.sigmar[end_idx:] + self.gamma_k[end_idx:]*self.rewards[self.idx].unsqueeze(0) 
                    G_new = self.sigmar[end_idx:] + self.gamma*self.gamma_k[end_idx:]*(self.next_values[self.idx]*mask).unsqueeze(0)
                    self.G_prev[end_idx:] = G_new.clone().detach()
                    self.gamma_k[end_idx:] = self.gamma*self.gamma_k[end_idx:]
                    Vt_new = (1-self.lam)*self.sigmaG[end_idx:] + self.lam_t[end_idx:]*G_new
                    self.sigmaG[end_idx:] = self.sigmaG[end_idx:] + self.lam_t[end_idx:]*G_new
                    self.lam_t[end_idx:] = self.lam_t[end_idx:]*self.lam
                    self.Vt_out[end_idx:] = self.Vt_out[end_idx:]*(1-self.mask_t[end_idx:]) + Vt_new*self.mask_t[end_idx:]
                    self.mask_t[end_idx:] = self.mask_t[end_idx:]*(self.grad_masks[self.idx].unsqueeze(0))

                if self.idx>0:
                    mask = self.masks[self.idx]
                    self.sigmar[start_idx:self.idx] = self.sigmar[start_idx:self.idx] + self.gamma_k[start_idx:self.idx]*self.rewards[self.idx].unsqueeze(0)
                    G_new = self.sigmar[start_idx:self.idx] + self.gamma*self.gamma_k[start_idx:self.idx]*(self.next_values[self.idx]*mask).unsqueeze(0)
                    self.G_prev[start_idx:self.idx] = G_new.clone().detach()
                    self.gamma_k[start_idx:self.idx] = self.gamma*self.gamma_k[start_idx:self.idx]
                    Vt_new = (1-self.lam)*self.sigmaG[start_idx:self.idx] + self.lam_t[start_idx:self.idx]*G_new
                    self.sigmaG[start_idx:self.idx] = self.sigmaG[start_idx:self.idx] + self.lam_t[start_idx:self.idx]*G_new
                    self.lam_t[start_idx:self.idx] = self.lam_t[start_idx:self.idx]*self.lam
                    self.Vt_out[start_idx:self.idx] = self.Vt_out[start_idx:self.idx]*(1-self.mask_t[start_idx:self.idx]) + Vt_new*self.mask_t[start_idx:self.idx]
                    self.mask_t[start_idx:self.idx] = self.mask_t[start_idx:self.idx]*(self.grad_masks[self.idx].unsqueeze(0))

            else:
                if self.idx > 0:
                    mask = self.masks[self.idx]
                    self.sigmar[:self.idx] = self.sigmar[:self.idx] + self.gamma_k[:self.idx]*self.rewards[self.idx].unsqueeze(0) 
                    G_new = self.sigmar[:self.idx] + self.gamma*self.gamma_k[:self.idx]*(self.next_values[self.idx]*mask).unsqueeze(0)
                    self.G_prev[:self.idx] = G_new.clone().detach()
                    self.gamma_k[:self.idx] = self.gamma*self.gamma_k[:self.idx]
                    Vt_new = (1-self.lam)*self.sigmaG[:self.idx] + self.lam_t[:self.idx]*G_new
                    self.sigmaG[:self.idx] = self.sigmaG[:self.idx] + self.lam_t[:self.idx]*G_new
                    self.lam_t[:self.idx] = self.lam_t[:self.idx]*self.lam
                    self.Vt_out[:self.idx] = self.Vt_out[:self.idx]*(1-self.mask_t[:self.idx]) + Vt_new*self.mask_t[:self.idx]
                    self.mask_t[:self.idx]= self.mask_t[:self.idx]*(self.grad_masks[self.idx].unsqueeze(0))

            ## Initializing for self.idx
            mask = self.masks[self.idx]
            self.sigmar[self.idx] = self.rewards[self.idx].clone()
            G_new = self.sigmar[self.idx] + self.gamma*self.next_values[self.idx]*mask
            self.G_prev[self.idx] = G_new.clone().detach()
            self.gamma_k[self.idx] = self.gamma
            self.sigmaG[self.idx] = G_new.clone().detach()
            Vt_new = G_new
            self.lam_t[self.idx] = self.lam
            self.Vt_out[self.idx] = Vt_new
            self.mask_t[self.idx] = self.grad_masks[self.idx].clone().detach()

        elif self.critic_method == 'gae-return':
            if self.full:
                # If the buffer is full, compute the start and end indices for the horizon to compute the gae-return
                # After exceeding the buffer length, the experiences wrap around to the beginning of the buffer
                if self.idx >= self.h-1:
                    start_idx = self.idx - self.h + 1
                else:
                    # Accounting for the wrap around of the buffer. We want to update end_idx:end and start_idx:self.idx
                    start_idx = 0
                    end_idx = self.idx - self.h + 1
                    mask = self.masks[self.idx]
                    delta_gamma_k = self.gamma_lam_k[end_idx:] * (self.rewards[self.idx] - self.next_values[self.idx-1] + self.gamma * self.next_values[self.idx] * mask).unsqueeze(0)
                    self.Vt_out[end_idx:] = self.Vt_out[end_idx:] + delta_gamma_k * self.mask_t[end_idx:]
                    self.gamma_lam_k[end_idx:] = self.gamma_lam_k[end_idx:] * self.gamma * self.lam
                    self.mask_t[end_idx:] = self.mask_t[end_idx:] * self.term_masks[self.idx].unsqueeze(0)
                if self.idx > 0:
                    mask = self.masks[self.idx]
                    delta_gamma_k = self.gamma_lam_k[start_idx:self.idx] * (self.rewards[self.idx] - self.next_values[self.idx-1] + self.gamma * self.next_values[self.idx] * mask).unsqueeze(0) 
                    self.Vt_out[start_idx:self.idx] = self.Vt_out[start_idx:self.idx] + delta_gamma_k * self.mask_t[start_idx:self.idx]
                    self.gamma_lam_k[start_idx:self.idx] = self.gamma_lam_k[start_idx:self.idx] * self.gamma * self.lam
                    self.mask_t[start_idx:self.idx] = self.mask_t[start_idx:self.idx] * self.term_masks[self.idx].unsqueeze(0)
            else:
                # If the buffer is not full, only need to update start_idx:self.idx
                start_idx = max(0, self.idx - self.h + 1)
                mask = self.masks[self.idx]
                delta_gamma_k = self.gamma_lam_k[start_idx:self.idx] * (self.rewards[self.idx] - self.next_values[self.idx-1] + self.gamma * self.next_values[self.idx] * mask).unsqueeze(0)  
                self.Vt_out[start_idx:self.idx] = self.Vt_out[start_idx:self.idx] + delta_gamma_k * self.mask_t[start_idx:self.idx]
                self.gamma_lam_k[start_idx:self.idx] = self.gamma_lam_k[start_idx:self.idx] * self.gamma * self.lam
                self.mask_t[start_idx:self.idx] = self.mask_t[start_idx:self.idx] * self.term_masks[self.idx].unsqueeze(0)

            # Update Vt_out, gamma_lam_k, and mask_t for the current timestep
            mask = self.masks[self.idx]
            delta_gamma_k = self.rewards[self.idx] + self.gamma * self.next_values[self.idx] * mask 
            self.Vt_out[self.idx] = delta_gamma_k
            self.gamma_lam_k[self.idx] = self.gamma * self.lam
            self.mask_t[self.idx] = self.term_masks[self.idx].clone().detach()

        else:
            raise NotImplementedError
            