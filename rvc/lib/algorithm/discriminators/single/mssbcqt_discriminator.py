import torch
import torch.nn as nn
#from torch.nn.utils import weight_norm
from torch.nn.utils.parametrizations import weight_norm

from torchaudio.transforms import Resample
from torch.utils.checkpoint import checkpoint

from rvc.train.utils import AttrDict

import typing
from typing import List, Tuple


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class DiscriminatorCQT(nn.Module):
    def __init__(self, cfg: AttrDict, hop_length: int, n_octaves: int, bins_per_octave: int, sample_rate: int, use_checkpointing: bool):
        super().__init__()
        self.cfg = cfg
        self.use_checkpointing = use_checkpointing

        self.filters = cfg["cqtd_filters"]
        self.max_filters = cfg["cqtd_max_filters"]
        self.filters_scale = cfg["cqtd_filters_scale"]
        self.kernel_size = (3, 9)
        self.dilations = cfg["cqtd_dilations"]
        self.stride = (1, 2)

        self.in_channels = cfg["cqtd_in_channels"]
        self.out_channels = cfg["cqtd_out_channels"]
        self.fs = sample_rate
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        from nnAudio import features
        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        # per-octave pre-convs
        self.conv_pres = nn.ModuleList([
            nn.Conv2d(
                self.in_channels * 2,
                self.in_channels * 2,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            )
            for _ in range(self.n_octaves)
        ])

        # main conv stack
        convs: List[nn.Module] = []
        convs.append(nn.Conv2d(
            self.in_channels * 2,
            self.filters,
            kernel_size=self.kernel_size,
            padding=self.get_2d_padding(self.kernel_size)
        ))
        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min((self.filters_scale ** (i + 1)) * self.filters, self.max_filters)
            convs.append(weight_norm(nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=(dilation, 1),
                padding=self.get_2d_padding(self.kernel_size, (dilation, 1)),
            )))
            in_chs = out_chs
        out_chs = min((self.filters_scale ** (len(self.dilations) + 1)) * self.filters, self.max_filters)
        convs.append(weight_norm(nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0]))
        )))
        self.convs = nn.ModuleList(convs)

        # post conv
        self.conv_post = weight_norm(nn.Conv2d(
            out_chs,
            self.out_channels,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0]))
        ))

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = Resample(orig_freq=self.fs, new_freq=self.fs * 2)
        self.cqtd_normalize_volume = cfg.get("cqtd_normalize_volume", False)
        if self.cqtd_normalize_volume:
            print("[INFO] cqtd_normalize_volume=True: applying DC offset removal & peak normalization.")

    def get_2d_padding(self, kernel_size: typing.Tuple[int, int], dilation: typing.Tuple[int, int] = (1, 1)):
        return (((kernel_size[0] - 1) * dilation[0]) // 2,
                ((kernel_size[1] - 1) * dilation[1]) // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Remove DC and normalize
        if self.cqtd_normalize_volume:
            x = x - x.mean(dim=-1, keepdims=True)
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        # Resample and CQT
        x = self.resample(x)
        z = self.cqt_transform(x)
        z_amp = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)
        z = torch.cat([z_amp, z_phase], dim=1).permute(0, 1, 3, 2)

        fmap: List[torch.Tensor] = []
        # Pre-convs per octave (checkpoint each conv individually)
        latent_parts = []
        for i, conv in enumerate(self.conv_pres):
            seg = z[:, :, :, i * self.bins_per_octave:(i + 1) * self.bins_per_octave]
            if self.training and self.use_checkpointing:
                out = checkpoint(conv, seg, use_reentrant=False)
            else:
                out = conv(seg)
            latent_parts.append(out)
        latent = torch.cat(latent_parts, dim=-1)

        # Main conv blocks
        for conv in self.convs:
            if self.training and self.use_checkpointing:
                latent = checkpoint(conv, latent, use_reentrant=False)
            else:
                latent = conv(latent)
            latent = self.activation(latent)
            fmap.append(latent)

        # Post conv
        if self.training and self.use_checkpointing:
            latent = checkpoint(self.conv_post, latent, use_reentrant=False)
        else:
            latent = self.conv_post(latent)

        return latent, fmap

class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, use_checkpointing, sample_rate, **kwargs):
        super().__init__()
        self.sample_rate = sample_rate
        self.use_checkpointing = use_checkpointing
        self.cfg = kwargs
        # defaults
        self.cfg.update({
            "cqtd_filters": self.cfg.get("cqtd_filters", 32),
            "cqtd_max_filters": self.cfg.get("cqtd_max_filters", 1024),
            "cqtd_filters_scale": self.cfg.get("cqtd_filters_scale", 1),
            "cqtd_dilations": self.cfg.get("cqtd_dilations", [1, 2, 4]),
            "cqtd_in_channels": self.cfg.get("cqtd_in_channels", 1),
            "cqtd_out_channels": self.cfg.get("cqtd_out_channels", 1),
            "cqtd_hop_lengths": self.cfg.get("cqtd_hop_lengths", [512, 256, 256]),
            "cqtd_n_octaves": self.cfg.get("cqtd_n_octaves", [9, 9, 9]),
            "cqtd_bins_per_octaves": self.cfg.get("cqtd_bins_per_octaves", [24, 36, 48]),
        })
        self.discriminators = nn.ModuleList([
            DiscriminatorCQT(
                self.cfg,
                hop_length=self.cfg["cqtd_hop_lengths"][i],
                n_octaves=self.cfg["cqtd_n_octaves"][i],
                bins_per_octave=self.cfg["cqtd_bins_per_octaves"][i],
                sample_rate=self.sample_rate,
                use_checkpointing=self.use_checkpointing
            ) for i in range(len(self.cfg["cqtd_hop_lengths"]))
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            if d.use_checkpointing:
                y_r, f_r = checkpoint(d.forward, y, use_reentrant=False)
                y_g, f_g = checkpoint(d.forward, y_hat, use_reentrant=False)
            else:
                y_r, f_r = d(y)
                y_g, f_g = d(y_hat)
            y_d_rs.append(y_r)
            fmap_rs.append(f_r)
            y_d_gs.append(y_g)
            fmap_gs.append(f_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
