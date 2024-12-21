import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

class Fixed_Heatmap:
    """
    Use pre-defined matrix as probability distribution
    """
    def __init__(self):
        super(Fixed_Heatmap, self).__init__()

    def sample(self):
        """
        Samples a point
        """
        return self.probs

    def log_probs(self, value):
        """
        return log(value) if value > 0
        """
        return (value > 0).type_as(value).log()

    def entropy(self):
        """
        return zeros
        """
        return torch.zeros(1)

    def forward(self, x):
        self.probs = F.softmax(x.view(x.size(0), -1), dim=1)\
                     .view_as(x)
        return self

    def __call__(self, x):
        return self.forward(x)

class Heatmap(nn.Module):
    """
    Generate Heatmap as probability distribution
    """
    def __init__(self):
        super(Heatmap, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.reset_counter = 0

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return HeatmapCategorical(logits=self.log_softmax(x))

class MultiHeatmap(nn.Module):
    """
    Generate multiple heatmaps as probability distributions.
    One for each robot.
    """
    def __init__(self):
        super(MultiHeatmap, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.reset_counter = 0

    def forward(self, x):
        # x shape: (batch_size, num_robots, map_width, map_height)
        x = x.view(x.size(0), x.size(1), -1)
        return MultiHeatmapCategorical(logits=self.log_softmax(x))

class DiagGaussian(nn.Module):
    """
    Diagonal Gaussian distribution.
    """
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class AddBias(nn.Module):
    """
    Add bias to input.
    """
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class MultiHeatmapCategorical(object):
    """
    Multiple categorical distributions for Heatmap.
    """
    def __init__(self, logits):
        self.logits = logits
        self.categoricals = [HeatmapCategorical(logits=logits[:, i]) for i in range(logits.size(1))]

    def mode(self):
        return torch.stack([categorical.mode() for categorical in self.categoricals], dim=1)

    def sample(self):
        return torch.stack([categorical.sample() for categorical in self.categoricals], dim=1)

    def log_probs(self, actions):
        return torch.stack([categorical.log_probs(actions[:, i]) 
                          for i, categorical in enumerate(self.categoricals)], dim=1)

    def entropy(self):
        return torch.stack([categorical.entropy() for categorical in self.categoricals], dim=1)

class HeatmapCategorical(object):
    """
    Categorical distribution for Heatmap.
    """
    def __init__(self, logits):
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)

    def mode(self):
        return self.probs.argmax(dim=-1)

    def sample(self):
        return torch.multinomial(self.probs, 1).squeeze(-1)

    def log_probs(self, actions):
        return self.logits.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    def entropy(self):
        p_log_p = self.logits * self.probs
        return -p_log_p.sum(-1)

class Categorical(nn.Module):
    """
    Categorical distribution for discrete action spaces
    """
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)

def init(module, weight_init, bias_init, gain=1):
    """
    Initialize module weights
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

FixedCategorical = torch.distributions.Categorical
log_prob_categorical = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_categorical(self, actions)

categorical_entropy = FixedCategorical.entropy
FixedCategorical.entropy = lambda self: categorical_entropy(self)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)