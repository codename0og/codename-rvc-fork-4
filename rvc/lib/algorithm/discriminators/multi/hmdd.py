# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

# DWT code borrow from https://github.com/LiQiufu/WaveSNet/blob/12cb9d24208c3d26917bf953618c30f0c6b0f03d/DWT_IDWT/DWT_IDWT_layer.py

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

import torch
import pywt
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

from torchaudio.transforms import Spectrogram, Resample

import typing
from typing import Optional, List, Union, Dict, Tuple

from rvc.train.utils import AttrDict

__all__ = ['DWT_1D']
Pad_Mode = ['constant', 'reflect', 'replicate', 'circular']

LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class DWT_1D(nn.Module):
    def __init__(self, pad_type='reflect', wavename='haar',
                 stride=2, in_channels=1, out_channels=None, groups=None,
                 kernel_size=None, trainable=False):

        super(DWT_1D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.rec_lo)
        band_high = torch.tensor(wavelet.rec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_low = self.filt_low[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        self.filter_high = self.filt_high[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        if torch.cuda.is_available():
            self.filter_low = self.filter_low.cuda()
            self.filter_high = self.filter_high.cuda()
        if self.trainable:
            self.filter_low = nn.Parameter(self.filter_low)
            self.filter_high = nn.Parameter(self.filter_high)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2]

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 3
        assert input.size()[1] == self.in_channels
        input = F.pad(input, pad=self.pad_sizes, mode=self.pad_type)
        return F.conv1d(input, self.filter_low.to(input.device), stride=self.stride, groups=self.groups), \
               F.conv1d(input, self.filter_high.to(input.device), stride=self.stride, groups=self.groups)


class DiscriminatorR(nn.Module):
    def __init__(self, cfg: AttrDict, resolution: List[List[int]]):
        super().__init__()
        self.cfg = cfg

        self.resolution = resolution
        assert len(self.resolution) == 3, f"MRD layer requires list with len=3, got {self.resolution}"

        self.lrelu_slope = 0.1
        self.d_mult = 1

        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        self.conv_post = weight_norm(
            nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        window = torch.hann_window(win_length, device=x.device)
        x = F.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            return_complex=True,
        )
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = kwargs
        self.resolutions = self.cfg["resolutions"]

        assert len(self.resolutions) == 3, \
            f"MRD requires list of list with len=3, each element having a list with len=3. Got {self.resolutions}"

        self.discriminators = nn.ModuleList([
            DiscriminatorR(
                self.cfg,
                resolution
            ) for resolution in self.resolutions
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
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
        x_d3 = self.dwt_conv3(
            torch.cat([x_d3_high1, x_d3_low1, x_d3_high2, x_d3_low2, x_d3_high3, x_d3_low3, x_d3_high4, x_d3_low4],
                      dim=1))
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
        for l in self.convs:
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
            i = i + 1
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class ResWiseMultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(ResWiseMultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
            DiscriminatorP(17),
            DiscriminatorP(23),
            DiscriminatorP(37),
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


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.dwt1d = DWT_1D()
        self.dwt_conv1 = norm_f(Conv1d(2, 128, 15, 1, padding=7))
        self.dwt_conv2 = norm_f(Conv1d(4, 128, 41, 2, padding=20))
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        # DWT 1
        x_d1_high1, x_d1_low1 = self.dwt1d(x)
        x_d1 = self.dwt_conv1(torch.cat([x_d1_high1, x_d1_low1], dim=1))

        # DWT 2
        x_d2_high1, x_d2_low1 = self.dwt1d(x_d1_high1)
        x_d2_high2, x_d2_low2 = self.dwt1d(x_d1_low1)
        x_d2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))

        i = 0
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
            if i == 0:
                x = torch.cat([x, x_d1], dim=2)
            if i == 1:
                x = torch.cat([x, x_d2], dim=2)
            i = i + 1
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class ResWiseMultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(ResWiseMultiScaleDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.dwt1d = DWT_1D()
        self.dwt_conv1 = norm_f(Conv1d(2, 1, 1))
        self.dwt_conv2 = norm_f(Conv1d(4, 1, 1))
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        # DWT 1
        y_hi, y_lo = self.dwt1d(y)
        y_1 = self.dwt_conv1(torch.cat([y_hi, y_lo], dim=1))
        x_d1_high1, x_d1_low1 = self.dwt1d(y_hat)
        y_hat_1 = self.dwt_conv1(torch.cat([x_d1_high1, x_d1_low1], dim=1))

        # DWT 2
        x_d2_high1, x_d2_low1 = self.dwt1d(y_hi)
        x_d2_high2, x_d2_low2 = self.dwt1d(y_lo)
        y_2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))

        x_d2_high1, x_d2_low1 = self.dwt1d(x_d1_high1)
        x_d2_high2, x_d2_low2 = self.dwt1d(x_d1_low1)
        y_hat_2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))

        for i, d in enumerate(self.discriminators):

            if i == 1:
                y = y_1
                y_hat = y_hat_1
            if i == 2:
                y = y_2
                y_hat = y_hat_2

            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class HolisticMultiDomainDiscriminator(nn.Module):
    """
    Holistic discriminator framework
    """
    def __init__(self, use_spectral_norm: bool = False, **multi_resolution_cfg):
        super(HolisticMultiDomainDiscriminator, self).__init__()

        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.multi_resolution_disc = MultiResolutionDiscriminator(**multi_resolution_cfg)

        self.period_disc = ResWiseMultiPeriodDiscriminator()
        self.scale_disc = ResWiseMultiScaleDiscriminator()

        self.discriminators = nn.ModuleList([self.multi_resolution_disc, self.period_disc, self.scale_disc])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, y_d_g, fmap_r, fmap_g = disc(y, y_hat)
            y_d_rs.extend(y_d_r)
            fmap_rs.extend(fmap_r)
            y_d_gs.extend(y_d_g)
            fmap_gs.extend(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
