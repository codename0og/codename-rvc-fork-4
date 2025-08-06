import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

from torchaudio.transforms import Spectrogram, Resample

from rvc.train.utils import AttrDict
import typing
from typing import Optional, List, Union, Dict, Tuple

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

