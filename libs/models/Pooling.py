import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.weight1 = Parameter(torch.Tensor(dh, din))
        self.weight2 = Parameter(torch.Tensor(1, dh))

    def forward(self, x: torch.Tensor, dim) -> torch.Tensor:
        # x : (Batch, F_dim, Time)
        # h : (Batch, Time, F_dim)
        h = x.transpose(1, 2)
        attn = F.relu(F.linear(h, self.weight1))
        attn = F.softmax(F.linear(attn, self.weight2), dim=1)

        mean = torch.bmm(x, attn)
        variance = (x - mean).pow(2).mean(dim=dim)
        mean = mean.squeeze(dim)

        mask = (variance <= self.eps).type(variance.dtype).to(variance.device)
        variance = (1.0 - mask) * variance + mask * self.eps
        stddev = variance.sqrt()

        # mean: (B, F_dim), stddev: (B, F_dim)
        pooling = torch.cat((mean, stddev), dim=-1)
        return pooling
