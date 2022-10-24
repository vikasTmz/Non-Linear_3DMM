import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.layers import ResnetBlockConv2d, EqualizedLR, ResnetBlockFC


class IdentityEncoder_ResnetFC(nn.Module):
    def __init__(self, disp_size, n_id=32, leaky=True):
        super().__init__()

        # Submodules
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.fc_dNid = nn.Linear(disp_size, n_id)

        self.block0 = ResnetBlockFC(n_id, actvn=self.actvn)
        self.block1 = ResnetBlockFC(n_id, actvn=self.actvn)
        self.block2 = ResnetBlockFC(n_id, actvn=self.actvn)
        self.block3 = ResnetBlockFC(n_id, actvn=None)
       
        # input dim is n_id * 3 as input displacement has three elements
        self.fc_mean = nn.Linear(n_id*3, n_id)
        self.fc_logstd = nn.Linear(n_id*3, n_id)

    def forward(self, neutral_geom, **kwargs):

        batch_size = neutral_geom.size(0)
        # fake_input = torch.ones(neutral_geom.size()).to(neutral_geom.device)
        out = self.fc_dNid(neutral_geom)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = out.view(batch_size, -1)

        mean = self.fc_mean(out)
        logstd = self.fc_logstd(out)

        return mean, logstd

class ExpressionEncoder_ResnetFC(nn.Module):
    def __init__(self, b_size, n_exp=256, leaky=True):
        super().__init__()

        # Submodules
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.fc_dNid = nn.Linear(b_size, n_exp)

        self.block0 = ResnetBlockFC(n_exp, actvn=self.actvn)
        self.block1 = ResnetBlockFC(n_exp, actvn=self.actvn)
        self.block2 = ResnetBlockFC(n_exp, actvn=self.actvn)
        self.block3 = ResnetBlockFC(n_exp, actvn=None)
       
        self.fc_mean = nn.Linear(n_exp, n_exp)
        self.fc_logstd = nn.Linear(n_exp, n_exp)

    def forward(self, blendweight, **kwargs):

        batch_size = blendweight.size(0)

        out = self.fc_dNid(blendweight)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = out.view(batch_size, -1)

        mean = self.fc_mean(out)
        logstd = self.fc_logstd(out)

        return mean, logstd
