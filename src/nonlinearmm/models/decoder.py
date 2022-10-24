import torch
import torch.nn as nn
import torch.nn.functional as F
from src import common
from src.layers import (
    ResnetBlockPointwise,
    ResnetBlockFC,
    EqualizedLR
)


class GeomDecoder_ResnetFC(nn.Module):
    def __init__(self, disp_size, n_id=32, n_exp=256, leaky=True):
        super().__init__()

        # Submodules
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.block0 = ResnetBlockFC(n_id + n_exp, actvn=self.actvn)
        self.block1 = ResnetBlockFC(n_id + n_exp, actvn=self.actvn)
        self.block2 = ResnetBlockFC(n_id + n_exp, actvn=self.actvn)
        self.block3 = ResnetBlockFC(n_id + n_exp, actvn=self.actvn)
       
        # output is disp_size*3 as output displacement has 3 elements
        self.fc_out = nn.Linear(n_id + n_exp, disp_size*3)

    def forward(self, z):
        batch_size = z.size(0)
        out = self.block0(z)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.fc_out(out)
        out = out.view(batch_size, 3, -1)
        # out = torch.sigmoid(out) # ToDo

        return out, None

class JointDecoder_ResnetFC(nn.Module):
    def __init__(self, disp_size, n_id=32, n_exp=256, leaky=True):
        super().__init__()

        # Submodules
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.block0 = ResnetBlockFC(n_id + n_exp, actvn=self.actvn)
        self.block1 = ResnetBlockFC(n_id + n_exp, actvn=self.actvn)
        self.block2 = ResnetBlockFC(n_id + n_exp, actvn=self.actvn)
        self.block3 = ResnetBlockFC(n_id + n_exp, actvn=self.actvn)
       
        # output is disp_size*3 as output displacement has 3 elements       
        self.fc_geom_out = nn.Linear(n_id + n_exp, disp_size*3)
        self.fc_albedo_out = nn.Linear(n_id + n_exp, disp_size*3)

    def forward(self, z):
        batch_size = z.size(0)
        out = self.block0(z)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        geom_disp = self.fc_geom_out(out)
        albedo_disp = self.fc_albedo_out(out)

        geom_disp = geom_disp.view(batch_size, 3, -1)
        albedo_disp = albedo_disp.view(batch_size, 3, -1)

        # albedo_disp = torch.sigmoid(albedo_disp)

        return geom_disp, albedo_disp
