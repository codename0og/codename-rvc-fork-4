import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from torch.nn import Conv1d, ConvTranspose1d

from torch.amp import autocast # guard
from torch.utils.checkpoint import checkpoint

import einops
import numpy as np

from rvc.lib.algorithm.residuals import ResBlock, ResBlock_Snake, ResBlock_Snake_Fused
from rvc.lib.algorithm.conformer.conformer import Conformer

from rvc.lib.algorithm.commons import init_weights
from rvc.lib.algorithm.conformer.stft import TorchSTFT



class SineGen(torch.nn.Module):
    """
    Definition of sine generator
    Args:
        samp_rate: sampling rate in Hz
        harmonic_num: number of harmonic overtones (default 0)
        sine_amp: amplitude of sine-wavefrom (default 0.1)
        noise_std: std of Gaussian noise (default 0.003)
        voiced_thoreshold: F0 threshold for U/V classification (default 0)
    """
    def __init__(
        self,
        samp_rate,
        upsample_scale,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        apply_modulo=False,
        double_precision=False,
        early_modulo=False
    ):
        super(SineGen, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale
        self.apply_modulo = apply_modulo
        self.double_precision = double_precision
        self.early_modulo = early_modulo

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], \
                              device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        rad_values = torch.nn.functional.interpolate(
            rad_values.transpose(1, 2), 
            scale_factor=1/self.upsample_scale, 
            mode="linear"
        ).transpose(1, 2)

        if not self.apply_modulo:
            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = torch.nn.functional.interpolate(
                phase.transpose(1, 2) * self.upsample_scale, 
                scale_factor=self.upsample_scale,
                mode="linear"
            ).transpose(1, 2)

            sines = torch.sin(phase)
        else:  # Modulo on fundamental only, harmonics = multiples without modulo
            if not self.early_modulo: # For late modulo style
                # Accumulate fundamental phase (first harmonic only), shape (batch, length)
                if self.double_precision:
                    # cast to double just for the sum …
                    fund_rad = rad_values[:, :, 0].double()
                    fund_phase_cum = torch.cumsum(fund_rad, dim=1)
                    fund_phase_cum = fund_phase_cum.float()
                else:
                    fund_rad = rad_values[:, :, 0]  # fundamental harmonic cycles per step
                    fund_phase_cum = torch.cumsum(fund_rad, dim=1)  # cumulative phase in cycles

                # interpolate upsampling if needed
                fund_phase_upsampled = torch.nn.functional.interpolate(
                    (fund_phase_cum * 2 * np.pi).unsqueeze(2).transpose(1, 2) * self.upsample_scale,
                    scale_factor=self.upsample_scale,
                    mode="linear"
                ).transpose(1, 2)  # shape (batch, length, 1), radians

                # wrap fundamental phase into [0, 2pi)
                fund_phase_wrapped = torch.fmod(fund_phase_upsampled, 2 * np.pi)

                # generate harmonics by multiplying fundamental phase (no modulo for harmonics)
                harmonics = torch.stack(
                    [fund_phase_wrapped[:, :, 0] * (k + 1) for k in range(self.dim)],
                    dim=2
                )  # shape (batch, length, dim)

                sines = torch.sin(harmonics)
            else:
                # --- EARLY‑MODULO PHASE ACCUMULATION (drift‑free, zero delay) ---
                # rad_values: [B, T, dim]; we only need the fundamental channel for carry
                fund_cycle = rad_values[:, :, 0].unsqueeze(-1)  # [B, T, 1] cycles per original sample

                # expand to per‑tick increments: [B, T, upsample_scale] cycles per tick
                idx = (torch.arange(self.upsample_scale, device=rad_values.device)
                    .view(1,1,-1).float() + 1)  # [1,1,U]
                phase_inc = fund_cycle * idx  # broadcast → per‑tick incremental cycles

                # compute the carry‑over remainder from each frame (last tick of each block)
                #    wrap it into [–0.5, +0.5) cycles so cumsum never drifts beyond one cycle
                rem = torch.fmod(phase_inc[:, :-1, -1:] + 0.5, 1.0) - 0.5  # [B, T-1, 1]

                # accumulate and wrap carry every frame
                cum_rem = torch.fmod(rem.cumsum(dim=1), 1.0)           # [B, T-1, 1]
                cum_rem = torch.nn.functional.pad(cum_rem, (0,0,1,0))  # pad to [B, T, 1]

                # add carry to each tick → true global phase per tick
                global_phase = phase_inc + cum_rem                    # [B, T, U] cycles

                # scale for harmonics and convert to radians
                k_idx = (torch.arange(1, self.dim+1, device=rad_values.device)
                        .view(1,1,-1).float())                          # [1,1,dim]
                harmonics = global_phase.unsqueeze(-1) * k_idx        # [B, T, U, dim]
                harmonics = harmonics * (2 * math.pi)                 # to radians

                # reshape to [B, T*U, dim] and take sin
                harmonics = harmonics.reshape(
                    rad_values.shape[0],
                    rad_values.shape[1] * self.upsample_scale,
                    self.dim
                )
                sines = torch.sin(harmonics)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
        f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            # Disabled mixed precision to avoid bf16 numerical errors in phase calculations
            with autocast(device_type="cuda", enabled=False):
                f0_buf = torch.zeros(
                    f0.shape[0],
                    f0.shape[1],
                    self.dim,
                    device=f0.device
                )
                # fundamental component
                fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

                # generate sine waveforms
                sine_waves = self._f02sine(fn) * self.sine_amp

                # generate uv signal
                uv = self._f02uv(f0)

                # noise: for unvoiced should be similar to sine_amp
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
                noise = noise_amp * torch.randn_like(sine_waves)

                # first: set the unvoiced part to 0 by uv
                # then: additive noise
                sine_waves = sine_waves * uv + noise
            return sine_waves, uv, noise

class SourceModuleHnNSF(torch.nn.Module):
    """
    Source Module for generating harmonic and noise components for audio synthesis.

    This module generates a harmonic source signal using sine waves and adds
    optional noise. It's often used in neural vocoders as a source of excitation.

    Args:
        sample_rate (int): Sampling rate of the audio in Hz.
        harmonic_num (int, optional): Number of harmonic overtones to generate above the fundamental frequency (F0). Defaults to 0.
        sine_amp (float, optional): Amplitude of the sine wave components. Defaults to 0.1.
        add_noise_std (float, optional): Standard deviation of the additive white Gaussian noise. Defaults to 0.003.
        voiced_threshold (float, optional): Threshold for the fundamental frequency (F0) to determine if a frame is voiced. If F0 is below this threshold, it's considered unvoiced. Defaults to 0.
    """
    def __init__(
        self,
        sample_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
        apply_modulo: bool = False,
        double_precision: bool = False,
        early_modulo: bool = False
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            samp_rate=sample_rate,
            upsample_scale=upsample_scale,
            harmonic_num=harmonic_num,
            sine_amp=sine_amp,
            noise_std=add_noise_std,
            voiced_threshold=voiced_threshold,
            apply_modulo=apply_modulo,
            double_precision=double_precision,
            early_modulo=early_modulo
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with autocast(device_type="cuda", enabled=False):
            with torch.no_grad():
                sine_wavs, uv, _ = self.l_sin_gen(x)
            sine_merge = self.l_tanh(self.l_linear(sine_wavs))

            # source for noise branch, in the same shape as uv
            noise = torch.randn_like(uv) * self.sine_amp / 3

            return sine_merge, noise, uv



class RingFormerGenerator(torch.nn.Module):
    def __init__(
        self,
        initial_channel, # 192
        resblock_kernel_sizes, # [3,7,11]
        resblock_dilation_sizes, # [[1,3,5], [1,3,5], [1,3,5]]
        upsample_rates, # [4, 4]
        upsample_initial_channel, # 512
        upsample_kernel_sizes, # [8, 8]
        gen_istft_n_fft, # 24khz: 60,  48khz: 120
        gen_istft_hop_size, # 24khz: 15, 48khz: 30
        gin_channels, # 256
        sr, # 24000, 48000, 
        harmonic_num = 8, # NFS Patch
        block_size_custom = 512,
        inplace_masking = False,
        checkpointing: bool = False,
    ):
        super(RingFormerGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        self.checkpointing = checkpointing

        #self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))


        '''
        Available ResBlock types:

            - ResBlock:  HiFi-Gan's 'ResBlock1' without any extras
            - ResBlock_Snake:  'ResBlock1' enriched with most basic Snake activation variant ( Borrowed from 'RiFornet' ) ; https://github.com/Respaired/RiFornet_Vocoder
            - ResBlock_Snake_Fused:  'ResBlock1' I modified by adding in Snake activation with triton fused kernel ( Forward, Backward ) ; https://github.com/falkaer/pytorch-snake
        '''
        ResBlock_Type = ResBlock_Snake_Fused


        self.m_source = SourceModuleHnNSF(
            sample_rate=sr,
            upsample_scale=math.prod(upsample_rates) * gen_istft_hop_size,
            harmonic_num=harmonic_num,
            voiced_threshold=0,
            apply_modulo=False,
            double_precision=False,
            early_modulo=False,
        )

        self.f0_upsamp = torch.nn.Upsample(
            scale_factor=math.prod(upsample_rates) * self.gen_istft_hop_size
        )

        self.noise_convs = nn.ModuleList()

        self.noise_res = nn.ModuleList()

        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2
                    )
                )
            )

            c_cur = upsample_initial_channel // (2 ** (i + 1))

            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                kernel = stride_f0 * 2 - stride_f0 % 2
                padding = 0 if stride_f0 == 1 else (kernel - stride_f0) // 2


                self.noise_convs.append(Conv1d(
                    self.gen_istft_n_fft + 2, c_cur, kernel_size=kernel, stride=stride_f0, padding=padding))

                self.noise_res.append(ResBlock_Type(c_cur, 7, [1, 3, 5]))
            else:
                self.noise_convs.append(Conv1d(self.gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(ResBlock_Type(c_cur, 11, [1, 3, 5]))


        self.alphas = nn.ParameterList()
        self.alphas.append(nn.Parameter(torch.ones(1, upsample_initial_channel, 1)))
        self.resblocks = nn.ModuleList()


        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.alphas.append(nn.Parameter(torch.ones(1, ch, 1)))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock_Type(ch, k, d))


        self.conformers = nn.ModuleList()
        self.post_n_fft = self.gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(128, self.post_n_fft + 2, 7, 1, padding=3))


        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** i)
            self.conformers.append(
                Conformer(
                    dim=ch,
                    depth=2,
                    dim_head=64,
                    heads=8,
                    ff_mult=4,
                    conv_expansion_factor = 2,
                    conv_kernel_size=31,
                    attn_dropout=0.1,
                    ff_dropout=0.1,
                    conv_dropout=0.1,
                    block_size=block_size_custom,
                )
            )


        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))

        self.stft = TorchSTFT(
            "cuda",
            filter_length=self.gen_istft_n_fft,
            hop_length=self.gen_istft_hop_size,
            win_length=self.gen_istft_n_fft
        )


        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)


    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None):

        debug_shapes = False

        if debug_shapes:
            print(f"[DEC forward] Input x shape (latent): {x.shape}")  # [batch, channels, time]
            print(f"[DEC forward] Input f0 shape: {f0.shape}")

        #f0, _, _ = self.F0_model(x.unsqueeze(1))

        if len(f0.shape) == 1:
            f0 = f0.unsqueeze(0)

        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        if debug_shapes:
            print(f"[DEC forward] Input f0 shape after f0_upsamp: {f0.shape}")

        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)

        if debug_shapes:
            print(f"[DEC forward] Shape of Latent {x.shape}") 
            print(f"[DEC forward] Shape of f0 {f0.shape}") 
            print(f"[DEC forward] Shape of har_source {har_source.shape}") 

        har_spec, har_phase = self.stft.transform(har_source) # NSF Patch
        har = torch.cat([har_spec, har_phase], dim=1) # NSF Patch

        x = self.conv_pre(x)

        if g is not None and self.cond is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):

            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
            x = einops.rearrange(x, 'b f t -> b t f')

            x = self.conformers[i](x)

            x = einops.rearrange(x, 'b t f -> b f t')

            # x = F.leaky_relu(x, LRELU_SLOPE)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source)


            if self.training and self.checkpointing:
                x = checkpoint(self.ups[i], x, use_reentrant=False)
                if i == self.num_upsamples - 1:
                    x = self.reflection_pad(x)
                x = x + x_source


                xs = None
                for j in range(self.num_kernels):
                    blk = self.resblocks[i * self.num_kernels + j]
                    cur = checkpoint(blk, x, use_reentrant=False)
                    xs = cur if xs is None else xs + cur
                x = xs / self.num_kernels
            else:
                x = self.ups[i](x)
                if i == self.num_upsamples - 1:
                    x = self.reflection_pad(x)

                x = x + x_source
    

                xs = None
                for j in range(self.num_kernels):
                    if xs is None:
                        xs = self.resblocks[i * self.num_kernels + j](x)
                    else:
                        xs += self.resblocks[i * self.num_kernels + j](x)
                x = xs / self.num_kernels

        # x = F.leaky_relu(x)

        x = x + (1 / self.alphas[i + 1]) * (torch.sin(self.alphas[i + 1] * x) ** 2)
        
        x = self.conv_post(x)

        spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])

        out = self.stft.inverse(spec, phase).to(x.device)

        return out, spec, phase

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    def __prepare_scriptable__(self):
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        # conv_pre, conv_post
        for hook in self.conv_pre._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                remove_weight_norm(self.conv_pre)
        for hook in self.conv_post._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                remove_weight_norm(self.conv_post)

        return self
