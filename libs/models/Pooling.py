import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


class StatsPooling(nn.Module):
    def __init__(self, eps=1e-12):
        super(StatsPooling, self).__init__()
        self.eps = eps

    def forward(self, x, dim=-1):
        # (Batch, F_dim, Time)
        mean = x.mean(dim=dim, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=dim)
        mean = mean.squeeze(dim)

        mask = (variance <= self.eps).type(variance.dtype).to(variance.device)
        variance = (1.0 - mask) * variance + mask * self.eps
        stddev = variance.sqrt()

        # mean: (B, F_dim), stddev: (B, F_dim)
        pooling = torch.cat((mean, stddev), dim=-1)
        return pooling


class AttnPooling(nn.Module):
    def __init__(self, din, dh=500, eps=1e-12):
        super(AttnPooling, self).__init__()
        self.eps = eps
        self.w1 = Parameter(torch.Tensor(dh, din))
        self.w2 = Parameter(torch.Tensor(1, dh))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, x, dim):
        # x : (Batch, F_dim, Time)
        attn = F.relu(F.linear(x.transpose(1, 2).contiguous(), self.w1))
        attn = F.softmax(F.linear(attn, self.w2), dim=1)

        mean = torch.bmm(x, attn)
        variance = (x - mean).pow(2)
        variance = torch.bmm(variance, attn).squeeze(dim)
        mean = mean.squeeze(dim)

        mask = (variance <= self.eps).type(variance.dtype).to(variance.device)
        variance = (1.0 - mask) * variance + mask * self.eps
        stddev = variance.sqrt()

        # mean: (B, F_dim), stddev: (B, F_dim)
        pooling = torch.cat((mean, stddev), dim=-1)
        return pooling
