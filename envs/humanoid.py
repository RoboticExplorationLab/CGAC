# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from envs.dflex_env import DFlexEnv
import math
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")

from utils import load_utils as lu
from utils import torch_utils as tu


class HumanoidEnv(DFlexEnv):

    def __init__(self, render=False, device='cuda:0', num_envs=4096, seed=0, episode_length=1000, no_grad=True, stochastic_init=False, MM_caching_frequency = 1):
        num_obs = 76
        num_act = 21

        super(HumanoidEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed, no_grad, render, device)

        self.stochastic_init = stochastic_init
        self.progress_buf_mask = torch.zeros_like(self.progress_buf)
        self.ep_lens = torch.zeros_like(self.progress_buf) + 10

        self.init_sim()

        # other parameters
        self.termination_height = 0.74
        self.motor_strengths = [
            200, 
            200, 
            200, 
            200, 
            200, 
            600, 
            400, 
            100, 
            100, 
            200, 
            200, 
            600, 
            400, 
            100,  
            100,
            100, 
            100, 
            200, 
            100, 
            100, 
            200]

        self.motor_scale = 0.35

        self.motor_strengths = tu.to_torch(self.motor_strengths, dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        self.action_penalty = -0.002
        self.joint_vel_obs_scaling = 0.1
        self.termination_tolerance = 0.1
        self.height_rew_scale = 10.0

        #-----------------------
        # set up Usd renderer
        if (self.visualize):
            self.stage = Usd.Stage.CreateNew("outputs/" + "Humanoid_" + str(self.num_envs) + ".usd")

            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0

    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1.0/60.0
        self.sim_substeps = 48
        self.sim_dt = self.dt

        self.ground = True

        self.num_joint_q = 28
        self.num_joint_qd = 27

        self.x_unit_tensor = tu.to_torch([1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch([0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch([0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        self.start_rot = df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)
        self.start_rotation = tu.to_torch(self.start_rot, device=self.device, requires_grad=False)

        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.y_unit_tensor.clone()
        self.heading_vec = self.x_unit_tensor.clone()
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = tu.to_torch([200.0, 0.0, 0.0], device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        self.start_pos = []

        if self.visualize:
            self.env_dist = 2.5
        else:
            self.env_dist = 0. # set to zero for training for numerical consistency

        start_height = 1.35

        asset_folder = os.path.join(os.path.dirname(__file__), 'assets')
        for i in range(self.num_environments):
            lu.parse_mjcf(os.path.join(asset_folder, "humanoid.xml"), self.builder,
                stiffness=5.0,
                damping=0.1,
                contact_ke=2.e+4,
                contact_kd=5.e+3,
                contact_kf=1.e+3,
                contact_mu=0.75,
                limit_ke=1.e+3,
                limit_kd=1.e+1,
                armature=0.007,
                load_stiffness=True,
                load_armature=True)

            # base transform
            start_pos_z = i*self.env_dist
            self.start_pos.append([0.0, start_height, start_pos_z])

            self.builder.joint_q[i*self.num_joint_q:i*self.num_joint_q + 3] = self.start_pos[-1]
            self.builder.joint_q[i*self.num_joint_q + 3:i*self.num_joint_q + 7] = self.start_rot

        num_q = int(len(self.builder.joint_q)/self.num_environments)
        num_qd = int(len(self.builder.joint_qd)/self.num_environments)
        print(num_q, num_qd)

        print("Start joint_q: ", self.builder.joint_q[0:num_q])

        self.start_joint_q = self.builder.joint_q[7:num_q].copy()
        self.start_joint_target = self.start_joint_q.copy()

        self.start_pos = tu.to_torch(self.start_pos, device=self.device)
        self.start_joint_q = tu.to_torch(self.start_joint_q, device=self.device)
        self.start_joint_target = tu.to_torch(self.start_joint_target, device=self.device)

        # finalize model
        self.model = self.builder.finalize(self.device)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=self.device)

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.state = self.model.state()

        num_act = int(len(self.state.joint_act) / self.num_environments) - 6
        print('num_act = ', num_act)

        if (self.model.ground):
            self.model.collide(self.state)

    def render(self, mode = 'human'):
        if self.visualize:
            self.render_time += self.dt
            self.renderer.update(self.state, self.render_time)

            if (self.num_frames == 1):
                try:
                    self.stage.Save()
                except:
                    print("USD save error")

                self.num_frames -= 1

    def step(self, actions, force_done_ids=None):
        actions = actions.view((self.num_envs, self.num_actions))

        # todo - make clip range a parameter
        actions = torch.clip(actions, -1., 1.)

        ##### an ugly fix for simulation nan values #### # reference: https://github.com/pytorch/pytorch/issues/15131
        def create_hook():
            def hook(grad):
                torch.nan_to_num(grad, 0.0, 0.0, 0.0, out = grad)
            return hook
        
        if self.state.joint_q.requires_grad:
            self.state.joint_q.register_hook(create_hook())
        if self.state.joint_qd.requires_grad:
            self.state.joint_qd.register_hook(create_hook())
        if actions.requires_grad:
            actions.register_hook(create_hook())
        #################################################

        self.actions = actions.clone()
        
        self.state.joint_act.view(self.num_envs, -1)[:, 6:] = actions * self.motor_scale * self.motor_strengths
        self.state = self.integrator.forward(self.model, self.state, self.sim_dt, self.sim_substeps, self.MM_caching_frequency)
        
        self.sim_time += self.sim_dt

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1
        self.progress_buf_mask *= 0

        self.calculateObservations()
        self.calculateReward()

        if force_done_ids is not None:
            self.reset_buf[force_done_ids] = 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        # if self.no_grad == False:
        self.obs_buf_before_reset = self.obs_buf.clone()
        self.extras = {
            'obs_before_reset': self.obs_buf_before_reset,
            'episode_end': self.termination_buf
            }

        if len(env_ids) > 0:
           self.reset(env_ids)

        if force_done_ids is not None:
            self.progress_buf_mask[force_done_ids] = self.episode_length

        self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def reset(self, env_ids = None, force_reset = True, eplenupdate=True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # clone the state to avoid gradient error
            self.state.joint_q = self.state.joint_q.clone().contiguous()
            self.state.joint_qd = self.state.joint_qd.clone().contiguous()

            # fixed start state
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] = self.start_pos[env_ids, :].clone()
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7] = self.start_rotation.unsqueeze(0).expand(len(env_ids),-1).clone()
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:] = self.start_joint_q.unsqueeze(0).expand(len(env_ids),-1).clone()
            self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.

            # randomization
            if self.stochastic_init:
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] = self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] + 0.1 * (torch.rand(size=(len(env_ids), 3), device=self.device) - 0.5) * 2.
                angle = (torch.rand(len(env_ids), device = self.device) - 0.5) * np.pi / 12.
                axis = torch.nn.functional.normalize(torch.rand((len(env_ids), 3), device = self.device) - 0.5)
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7] = tu.quat_mul(self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7], tu.quat_from_angle_axis(angle, axis))
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:] = self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:] + 0.2 * (torch.rand(size=(len(env_ids), self.num_joint_q - 7), device = self.device) - 0.5) * 2.
                self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.5 * (torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5)

            # clear action
            self.actions = self.actions.clone()
            self.actions[env_ids, :] = torch.zeros((len(env_ids), self.num_actions), device = self.device, dtype = torch.float)

            self.progress_buf_mask[env_ids] = self.progress_buf[env_ids].clone()
            if eplenupdate:
                self.ep_lens = torch.cat([self.progress_buf[env_ids].clone(), self.ep_lens], dim=0)[:200]
            self.progress_buf[env_ids] = 0

            self.calculateObservations()

        return self.obs_buf
    
    '''
    cut off the gradient from the current state to previous states
    '''
    def clear_grad(self, checkpoint = None):
        with torch.no_grad():
            if checkpoint is None:
                checkpoint = {}
                checkpoint['joint_q'] = self.state.joint_q.clone()
                checkpoint['joint_qd'] = self.state.joint_qd.clone()
                checkpoint['actions'] = self.actions.clone()
                checkpoint['progress_buf'] = self.progress_buf.clone()

            current_joint_q = checkpoint['joint_q'].clone()
            current_joint_qd = checkpoint['joint_qd'].clone()
            self.state = self.model.state()
            self.state.joint_q = current_joint_q
            self.state.joint_qd = current_joint_qd
            self.actions = checkpoint['actions'].clone()
            self.progress_buf = checkpoint['progress_buf'].clone()
        self.state.joint_q.requires_grad_(True)
        self.state.joint_qd.requires_grad_(True)
    '''
    This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and it returns the observation vectors
    '''
    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()

        return self.obs_buf

    def get_checkpoint(self):
        checkpoint = {}
        checkpoint['joint_q'] = self.state.joint_q.clone()
        checkpoint['joint_qd'] = self.state.joint_qd.clone()
        checkpoint['actions'] = self.actions.clone()
        checkpoint['progress_buf'] = self.progress_buf.clone()

        return checkpoint

    def calculateObservations(self, state=None, actions=None):
        update_self_obs = False
        if state is None:
            state = self.state
            actions = self.actions
            update_self_obs = True
        torso_pos = state.joint_q.view(self.num_envs, -1)[:, 0:3]
        torso_rot = state.joint_q.view(self.num_envs, -1)[:, 3:7]
        lin_vel = state.joint_qd.view(self.num_envs, -1)[:, 3:6]
        ang_vel = state.joint_qd.view(self.num_envs, -1)[:, 0:3]

        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin_vel = lin_vel - torch.cross(torso_pos, ang_vel, dim = -1)

        to_target = self.targets + self.start_pos - torso_pos
        to_target[:, 1] = 0.0
        
        target_dirs = tu.normalize(to_target)
        torso_quat = tu.quat_mul(torso_rot, self.inv_start_rot)

        up_vec = tu.quat_rotate(torso_quat, self.basis_vec1)
        heading_vec = tu.quat_rotate(torso_quat, self.basis_vec0)

        obs_buf = torch.cat([torso_pos[:, 1:2], # 0
                                torso_rot, # 1:5
                                lin_vel, # 5:8
                                ang_vel, # 8:11
                                state.joint_q.view(self.num_envs, -1)[:, 7:], # 11:32
                                self.joint_vel_obs_scaling * state.joint_qd.view(self.num_envs, -1)[:, 6:], # 32:53
                                up_vec[:, 1:2], # 53:54
                                (heading_vec * target_dirs).sum(dim = -1).unsqueeze(-1), # 54:55
                                actions.clone()], # 55:76
                                dim = -1)
        if update_self_obs:
            self.obs_buf = obs_buf

        return obs_buf

    def calculateObservationsEx(self, state=None, actions=None):
        update_self_obs = False
        bsz = actions.shape[0]
        torso_pos = state.joint_q.view(bsz, -1)[:, 0:3]
        torso_rot = state.joint_q.view(bsz, -1)[:, 3:7]
        lin_vel = state.joint_qd.view(bsz, -1)[:, 3:6]
        ang_vel = state.joint_qd.view(bsz, -1)[:, 0:3]

        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin_vel = lin_vel - torch.cross(torso_pos, ang_vel, dim = -1)

        to_target = self.targets[:1] + self.start_pos[:1] - torso_pos
        to_target[:, 1] = 0.0
        
        target_dirs = tu.normalize(to_target)
        torso_quat = tu.quat_mul(torso_rot, self.inv_start_rot[:1].expand(bsz, -1))

        up_vec = tu.quat_rotate(torso_quat, self.basis_vec1[:1].expand(bsz, -1))
        heading_vec = tu.quat_rotate(torso_quat, self.basis_vec0[:1].expand(bsz, -1))

        obs_buf = torch.cat([torso_pos[:, 1:2], # 0
                                torso_rot, # 1:5
                                lin_vel, # 5:8
                                ang_vel, # 8:11
                                state.joint_q.view(bsz, -1)[:, 7:], # 11:32
                                self.joint_vel_obs_scaling * state.joint_qd.view(bsz, -1)[:, 6:], # 32:53
                                up_vec[:, 1:2], # 53:54
                                (heading_vec * target_dirs).sum(dim = -1).unsqueeze(-1), # 54:55
                                actions.clone()], # 55:76
                                dim = -1)


        return obs_buf
        
    def calculateObservationsExt(self, state_joint, obs):
        actions = obs[:, -self.actions.shape[-1]:]
        bsz = obs.shape[0]
        joint_q_shape = self.state.joint_q.view(self.num_envs, -1).shape[-1]
        joint_qd_shape = self.state.joint_qd.view(self.num_envs, -1).shape[-1]
        joint_q = state_joint[:, :joint_q_shape]
        joint_qd = state_joint[:, joint_q_shape:]
        torso_pos = joint_q[:, 0:3]
        torso_rot = joint_q[:, 3:7]
        lin_vel = joint_qd[:, 3:6]
        ang_vel = joint_qd[:, 0:3]

        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin_vel = lin_vel - torch.cross(torso_pos, ang_vel, dim = -1)

        to_target = self.targets[:1] + self.start_pos[:1] - torso_pos
        to_target[:, 1] = 0.0
        
        target_dirs = tu.normalize(to_target)
        torso_quat = tu.quat_mul(torso_rot, self.inv_start_rot[:1].expand(bsz, -1))

        up_vec = tu.quat_rotate(torso_quat, self.basis_vec1[:1].expand(bsz, -1))
        heading_vec = tu.quat_rotate(torso_quat, self.basis_vec0[:1].expand(bsz, -1))

        obs_buf = torch.cat([torso_pos[:, 1:2], # 0
                                torso_rot, # 1:5
                                lin_vel, # 5:8
                                ang_vel, # 8:11
                                joint_q[:, 7:], # 11:32
                                self.joint_vel_obs_scaling * joint_qd[:, 6:], # 32:53
                                up_vec[:, 1:2], # 53:54
                                (heading_vec * target_dirs).sum(dim = -1).unsqueeze(-1), # 54:55
                                actions.clone()], # 55:76
                                dim = -1)
        # if update_self_obs:
        #     self.obs_buf = obs_buf

        return obs_buf

    def calculateReward(self):
        up_reward = 0.1 * self.obs_buf[:, 53]
        heading_reward = self.obs_buf[:, 54]

        height_diff = self.obs_buf[:, 0] - (self.termination_height + self.termination_tolerance)
        height_reward = torch.clip(height_diff, -1.0, self.termination_tolerance)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward)
        height_reward = torch.where(height_reward > 0.0, self.height_rew_scale * height_reward, height_reward)
        
        progress_reward = self.obs_buf[:, 5]

        self.rew_buf = progress_reward + up_reward + heading_reward + height_reward + torch.sum(self.actions ** 2, dim = -1) * self.action_penalty

        # reset agents
        self.reset_buf = torch.where(self.obs_buf[:, 0] < self.termination_height, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf > self.episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # an ugly fix for simulation nan values
        nan_masks = torch.logical_or(torch.isnan(self.obs_buf).sum(-1) > 0, torch.logical_or(torch.isnan(self.state.joint_q.view(self.num_environments, -1)).sum(-1) > 0, torch.isnan(self.state.joint_qd.view(self.num_environments, -1)).sum(-1) > 0))
        inf_masks = torch.logical_or(torch.isinf(self.obs_buf).sum(-1) > 0, torch.logical_or(torch.isinf(self.state.joint_q.view(self.num_environments, -1)).sum(-1) > 0, torch.isinf(self.state.joint_qd.view(self.num_environments, -1)).sum(-1) > 0))
        invalid_value_masks = torch.logical_or((torch.abs(self.state.joint_q.view(self.num_environments, -1)) > 1e6).sum(-1) > 0,
                                                (torch.abs(self.state.joint_qd.view(self.num_environments, -1)) > 1e6).sum(-1) > 0)   
        invalid_masks = torch.logical_or(invalid_value_masks, torch.logical_or(nan_masks, inf_masks))
        
        self.reset_buf = torch.where(invalid_masks, torch.ones_like(self.reset_buf), self.reset_buf)
    
        self.invalid_masks = invalid_masks
        self.rew_buf[invalid_masks] = 0.

    def calculateRewardEx(self, obs_buf, actions):
        up_reward = 0.1 * obs_buf[:, 53]
        heading_reward = obs_buf[:, 54]

        height_diff = obs_buf[:, 0] - (self.termination_height + self.termination_tolerance)
        height_reward = torch.clip(height_diff, -1.0, self.termination_tolerance)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward)
        height_reward = torch.where(height_reward > 0.0, self.height_rew_scale * height_reward, height_reward)
        
        progress_reward = obs_buf[:, 5]

        rew_buf = progress_reward + up_reward + heading_reward + height_reward + torch.sum(actions ** 2, dim = -1) * self.action_penalty

        rew_buf[invalid_masks] = 0.
        return rew_buf