import math
import torch
import torch.nn as nn
import numpy as np

def create_log_gaussian(mean, log_std, t):
    """Creates a log probability for a Gaussian distribution."""
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp."""
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def rand_sample(act_space, bsz):
    return torch.rand((bsz,act_space.shape[0]))*(act_space.high-act_space.low) + act_space.low


def grad_norm(params):
    """Computes the norm of gradients for a group of parameters."""
    grad_norm = 0.
    for p in params:
        if p.grad is not None:
            grad_norm += torch.sum(p.grad ** 2)
    return torch.sqrt(grad_norm)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


'''
updates statistic from a full data
'''
class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        print('RunningMeanStd: ', insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0,2,3]
            if len(self.insize) == 2:
                self.axis = [0,2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0] 
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype = torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype = torch.float64))
        self.register_buffer("count", torch.ones((), dtype = torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def update(self, input):
            mean = input.mean(self.axis) # along channel axis
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, 
                                                    mean, var, input.size()[0] )

    def forward(self, input, unnorm=False):

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.detach().view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.detach().view([1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.detach().view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.detach().view([1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_mean.detach().view([1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.detach().view([1, self.insize[0]]).expand_as(input)        
        else:
            current_mean = self.running_mean.detach()
            current_var = self.running_var.detach()
        # get output

        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            if self.norm_only:
                y = input/ torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y

'''
updates statistic from a full data
'''
class RunningGradMag(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False, const=1):
        super(RunningGradMag, self).__init__()
        print('RunningGradMag: ', insize)
        self.insize = insize
        self.epsilon = epsilon
        self.const = const

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0,2,3]
            if len(self.insize) == 2:
                self.axis = [0,2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0] 
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_absmean", torch.ones(in_size, dtype = torch.float64))
        self.register_buffer("count", torch.ones((), dtype = torch.float64))

    def _update_mean_var_count_from_moments(self, mean, count, batch_mean, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        new_count = tot_count
        return new_mean, new_count

    def update(self, input):
        mean = input.abs().mean(self.axis) # along channel axis
        self.running_absmean, self.count = self._update_mean_var_count_from_moments(self.running_absmean, self.count, 
                                                mean, input.size()[0] )

    def forward(self, input, unnorm=False):

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_absmean.detach().view([1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_absmean.detach().view([1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_absmean.detach().view([1, self.insize[0]]).expand_as(input)
        else:
            current_mean = self.running_absmean.detach()
        # get output

        if unnorm:
            y = input/self.const#(current_mean.float())
        else:
            y = input*self.const#(current_mean.float())
        return y

class RunningMeanStdObs(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        assert(insize is dict)
        super(RunningMeanStdObs, self).__init__()
        self.running_mean_std = nn.ModuleDict({
            k : RunningMeanStd(v, epsilon, per_channel, norm_only) for k,v in insize.items()
        })
    
    def forward(self, input, unnorm=False):
        res = {k : self.running_mean_std(v, unnorm) for k,v in input.items()}
        return res

class DummyRMS(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(DummyRMS, self).__init__()
        print('DummyRMS: ', insize)
        self.insize = insize
        self.epsilon = epsilon

    def forward(self, input, unnorm=False):
        return input

    def update(self, input):
        return None