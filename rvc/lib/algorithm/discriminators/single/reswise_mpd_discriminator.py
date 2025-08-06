import torch
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm

from rvc.lib.algorithm.custom_discriminators.freegan_disc_modules.dwt import DWT_1D


LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, use_checkpointing=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_checkpointing = use_checkpointing
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        # Initialize the DWT and convolutions
        self.dwt1d = DWT_1D()
        self.dwt_conv1 = norm_f(Conv1d(2, 1, 1))
        self.dwt_proj1 = norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))
        self.dwt_conv2 = norm_f(Conv1d(4, 1, 1))
        self.dwt_proj2 = norm_f(Conv2d(1, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))
        self.dwt_conv3 = norm_f(Conv1d(8, 1, 1))
        self.dwt_proj3 = norm_f(Conv2d(1, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))

        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])

        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # DWT 1
        x_d1_high1, x_d1_low1 = self.dwt1d(x)

        # Checkpoint DWT Conv1: Store input and recompute during backpropagation
        if self.use_checkpointing:
            x_d1 = checkpoint(self.dwt_conv1, torch.cat([x_d1_high1, x_d1_low1], dim=1), use_reentrant=True)
        else:
            x_d1 = self.dwt_conv1(torch.cat([x_d1_high1, x_d1_low1], dim=1))
        # 1d to 2d
        b, c, t = x_d1.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x_d1 = F.pad(x_d1, (0, n_pad), "reflect")
            t = t + n_pad
        x_d1 = x_d1.view(b, c, t // self.period, self.period)

        x_d1 = self.dwt_proj1(x_d1)

        # DWT 2
        x_d2_high1, x_d2_low1 = self.dwt1d(x_d1_high1)
        x_d2_high2, x_d2_low2 = self.dwt1d(x_d1_low1)


        if self.use_checkpointing:
            x_d2 = checkpoint(self.dwt_conv2, torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1), use_reentrant=True)
        else:
            x_d2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))

        # 1d to 2d
        b, c, t = x_d2.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x_d2 = F.pad(x_d2, (0, n_pad), "reflect")
            t = t + n_pad
        x_d2 = x_d2.view(b, c, t // self.period, self.period)

        x_d2 = self.dwt_proj2(x_d2)

        # DWT 3
        x_d3_high1, x_d3_low1 = self.dwt1d(x_d2_high1)
        x_d3_high2, x_d3_low2 = self.dwt1d(x_d2_low1)
        x_d3_high3, x_d3_low3 = self.dwt1d(x_d2_high2)
        x_d3_high4, x_d3_low4 = self.dwt1d(x_d2_low2)

        if self.use_checkpointing:
            x_d3 = checkpoint(self.dwt_conv3, torch.cat([x_d3_high1, x_d3_low1, x_d3_high2, x_d3_low2, x_d3_high3, x_d3_low3, x_d3_high4, x_d3_low4], dim=1), use_reentrant=True)
        else:
            x_d3 = self.dwt_conv3(torch.cat([x_d3_high1, x_d3_low1, x_d3_high2, x_d3_low2, x_d3_high3, x_d3_low3, x_d3_high4, x_d3_low4], dim=1))
        # 1d to 2d
        b, c, t = x_d3.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x_d3 = F.pad(x_d3, (0, n_pad), "reflect")
            t = t + n_pad
        x_d3 = x_d3.view(b, c, t // self.period, self.period)

        x_d3 = self.dwt_proj3(x_d3)

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        i = 0
        for i, l in enumerate(self.convs):
            if self.use_checkpointing:
                x = checkpoint(l, x, use_reentrant=True)  # Apply checkpointing here
            else:
                x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

            if i == 0:
                x = torch.cat([x, x_d1], dim=2)
            elif i == 1:
                x = torch.cat([x, x_d2], dim=2)
            elif i == 2:
                x = torch.cat([x, x_d3], dim=2)
            else:
                x = x
        if self.use_checkpointing:
            x = checkpoint(self.conv_post, x, use_reentrant=True)  # Apply checkpointing here
        else:
            x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class ResWiseMultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_checkpointing=False):
        super(ResWiseMultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
            DiscriminatorP(17), #
            DiscriminatorP(23), #
            DiscriminatorP(37), #
        ])
    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs