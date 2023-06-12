import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = 1e-6
activations_dict = {'elu': nn.ELU(),
                    'relu': nn.ReLU(),
                    'tanh': nn.Tanh()}

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, args):
        super(QNetwork, self).__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim, hidden_dim]
        layer_sizes = [num_inputs + num_actions,]+hidden_dim
        activation = activations_dict[args.critic_act]
        # Q1 architecture
        layers1 = []
        for i in range(len(layer_sizes)-1):
            layers1.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers1.append(activation) 
            layers1.append(nn.Identity())
        self.layers1 = nn.Sequential(*layers1)
        self.layer_out1 = nn.Linear(layer_sizes[-1], 1)

        # Q2 architecture
        layers2 = []
        for i in range(len(layer_sizes)-1):
            layers2.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers2.append(activation)
            layers2.append(nn.Identity())
        self.layers2 = nn.Sequential(*layers2)
        self.layer_out2 = nn.Linear(layer_sizes[-1], 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = self.layers1(xu)
        x1 = self.layer_out1(x1)

        x2 = self.layers2(xu)
        x2 = self.layer_out2(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, args, action_space=None, const_std=False):
        super(GaussianPolicy, self).__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim, hidden_dim]
        layer_sizes = [num_inputs,]+hidden_dim
        activation = activations_dict[args.actor_act]

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(activation)
            layers.append(nn.Identity())

        self.layers = nn.Sequential(*layers)
        self.mean_linear = nn.Linear(layer_sizes[-1], num_actions)

        self.const_std = const_std
        if const_std:
            logstd = -1.0#'actor_logstd_init'
            self.logstd = torch.nn.Parameter(torch.ones(num_actions, dtype=torch.float32) * logstd)
        else:
            self.log_std_linear = nn.Linear(layer_sizes[-1], num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = self.layers(state)
        mean = self.mean_linear(x)
        if self.const_std:
            log_std = self.logstd 
        else:
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def log_prob(self, state, action, x_t):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        y_t = (action - self.action_bias)/self.action_scale
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob

    def sample(self, state, netid=1, with_xt=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        if with_xt:
            return action, log_prob, mean, x_t
        else:
            return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim, hidden_dim]
        layer_sizes = [num_inputs,]+hidden_dim
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.mean = nn.Linear(layer_sizes[-1], num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = self.layers(state)
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
