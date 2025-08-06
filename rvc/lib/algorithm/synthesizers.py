import torch
from typing import Optional

# Generators
from rvc.lib.algorithm.generators.hifigan_mrf import HiFiGANMRFGenerator
from rvc.lib.algorithm.generators.hifigan_nsf import HiFiGANNSFGenerator
from rvc.lib.algorithm.generators.hifigan import HiFiGANGenerator
from rvc.lib.algorithm.generators.refinegan import RefineGANGenerator
from rvc.lib.algorithm.generators.ringformer import RingFormerGenerator #
#from rvc.lib.algorithm.generators.ringformer_nof0 import RingFormerGenerator_nof0 #

from rvc.lib.algorithm.commons import slice_segments, rand_slice_segments
from rvc.lib.algorithm.residuals import ResidualCouplingBlock
from rvc.lib.algorithm.encoders import TextEncoder, PosteriorEncoder

debug_shapes = False

class Synthesizer(torch.nn.Module):
    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool,
        text_enc_hidden_dim: int = 768,
        vocoder: str = "HiFi-GAN",
        randomized: bool = True,
        checkpointing: bool = False,
        gen_istft_n_fft: int = 120,
        gen_istft_hop_size: int = 30,
        **kwargs,
    ):
        super().__init__()
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        self.segment_size = segment_size
        self.use_f0 = use_f0
        self.randomized = randomized
        self.is_ringformer = vocoder == "RingFormer"

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            text_enc_hidden_dim,
            f0=use_f0,
        )
        if use_f0:
            if vocoder == "MRF HiFi-GAN":
                self.dec = HiFiGANMRFGenerator(
                    in_channel=inter_channels,
                    upsample_initial_channel=upsample_initial_channel,
                    upsample_rates=upsample_rates,
                    upsample_kernel_sizes=upsample_kernel_sizes,
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilations=resblock_dilation_sizes,
                    gin_channels=gin_channels,
                    sample_rate=sr,
                    harmonic_num=8,
                    checkpointing=checkpointing,
                )
                print("    ██████  Vocoder: NSF-HiFi-GAN ( MRF VARIANT )         ██████")
            elif vocoder == "RefineGAN":
                self.dec = RefineGANGenerator(
                    sample_rate=sr,
                    downsample_rates=upsample_rates[::-1],
                    upsample_rates=upsample_rates,
                    start_channels=16,
                    num_mels=inter_channels,
                    checkpointing=checkpointing,
                )
                print("    ██████  Vocoder: RefineGAN                            ██████")
            elif vocoder == "RingFormer":
                self.dec = RingFormerGenerator(
                    initial_channel=inter_channels,
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilation_sizes=resblock_dilation_sizes,
                    upsample_rates=upsample_rates,
                    upsample_initial_channel=upsample_initial_channel,
                    upsample_kernel_sizes=upsample_kernel_sizes,
                    gen_istft_n_fft=gen_istft_n_fft,
                    gen_istft_hop_size=gen_istft_hop_size,
                    gin_channels=gin_channels,
                    sr=sr,
                    checkpointing=checkpointing,
                )
                print("    ██████  Vocoder: RingFormer                           ██████")
            else:
                self.dec = HiFiGANNSFGenerator(
                    inter_channels,
                    resblock_kernel_sizes,
                    resblock_dilation_sizes,
                    upsample_rates,
                    upsample_initial_channel,
                    upsample_kernel_sizes,
                    gin_channels=gin_channels,
                    sr=sr,
                    checkpointing=checkpointing,
                )
                print("    ██████  Vocoder: NSF-HiFi-GAN.                        ██████")
        else:
            if vocoder == "MRF HiFi-GAN":
                print("MRF HiFi-GAN does not support training without pitch guidance.")
                self.dec = None
            elif vocoder == "RefineGAN":
                print("RefineGAN does not support training without pitch guidance.")
                self.dec = None
            elif vocoder == "RingFormer":
                print("RingFormer does not support training without pitch guidance.")
                self.dec = None
            else:
                self.dec = HiFiGANGenerator(
                    inter_channels,
                    resblock_kernel_sizes,
                    resblock_dilation_sizes,
                    upsample_rates,
                    upsample_initial_channel,
                    upsample_kernel_sizes,
                    gin_channels=gin_channels,
                    checkpointing=checkpointing,
                )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            3,
            gin_channels=gin_channels,
        )
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)

    def _remove_weight_norm_from(self, module):
        """Utility to remove weight normalization from a module."""
        for hook in module._forward_pre_hooks.values():
            if getattr(hook, "__class__", None).__name__ == "WeightNorm":
                torch.nn.utils.remove_weight_norm(module)

    def remove_weight_norm(self):
        """Removes weight normalization from the model."""
        for module in [self.dec, self.flow, self.enc_q]:
            self._remove_weight_norm_from(module)

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        spec: Optional[torch.Tensor] = None, # y
        spec_lengths: Optional[torch.Tensor] = None, # y_lengths
        ds: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the model.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            pitchf (torch.Tensor, optional): Fine-grained pitch sequence.
            spek (torch.Tensor, optional): Target spectrogram.  - y
            spek_lengths (torch.Tensor, optional): Lengths of the target spectrograms. - y_lengths
            ds (torch.Tensor, optional): Speaker embedding.
        """
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        if spec is not None:
            if debug_shapes:
                print(f"[DEBUG PRE-DECODER] spec_lengths shape: {spec_lengths}")
                print(f"[DEBUG PRE-DECODER] spec shape: {spec.shape}")

            z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)

            if debug_shapes:
                print(f"[DEBUG PRE-DECODER] z shape: {z.shape}")

            z_p = self.flow(z, spec_mask, g=g)


            if not self.is_ringformer: # For HiFi-Gan, MRF-HiFi-Gan and RefineGan training
            
                if self.randomized:
                    z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
                    if debug_shapes:
                        print(f"[DEBUG PRE-DECODER] z_slice shape: {z_slice.shape}")
                        print(f"[DEBUG PRE-DECODER] ids_slice: {ids_slice}")

                    if self.use_f0:
                        if debug_shapes:
                            print(f"[DEBUG] pitchf shape before slicing: {pitchf.shape}")

                        pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                        if debug_shapes:
                            print(f"[DEBUG PRE-DECODER] pitchf shape after slicing: {pitchf.shape}")

                        o = self.dec(z_slice, pitchf, g=g)
                    else:
                        o = self.dec(z_slice, g=g)

                    return o, ids_slice, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

                else:
                    if self.use_f0:
                        o = self.dec(z, pitchf, g=g)
                    else:
                        o = self.dec(z, g=g)

                    return o, None, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

            else: # For RingFormer training

                if self.randomized: # If using the slicing mechanism
                    z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
                    if debug_shapes:
                        print(f"[DEBUG PRE-DECODER] z_slice shape: {z_slice.shape}")
                        print(f"[DEBUG PRE-DECODER] ids_slice: {ids_slice}")

                        print(f"[DEBUG] pitchf shape before slicing: {pitchf.shape}")

                    pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                    if debug_shapes:
                        print(f"[DEBUG PRE-DECODER] pitchf shape after slicing: {pitchf.shape}")

                    o, spec, phase = self.dec(z_slice, pitchf, g=g) # f0 output

                    return o, ids_slice, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (spec, phase)

                else:
                    o, spec, phase = self.dec(z, pitchf, g=g) # f0 output
                    #o, spec, phase = self.dec(z, g=g) # f0 output

                    return o, None, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (spec, phase)
        else:
            print(" NONE SPEC ")
            return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        rate: Optional[torch.Tensor] = None,
    ):
        """
        Inference of the model.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            nsff0 (torch.Tensor, optional): Fine-grained pitch sequence.
            sid (torch.Tensor): Speaker embedding.
            rate (torch.Tensor, optional): Rate for time-stretching.
        """
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask

        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]

            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head:]

        z = self.flow(z_p, x_mask, g=g, reverse=True)

        #print("[INFER] pitchf:", nsff0.shape, nsff0.min(), nsff0.max())
        if not self.is_ringformer: # Non RingFormer
            o = (self.dec(z * x_mask, nsff0, g=g) if self.use_f0 else self.dec(z * x_mask, g=g))
        else:
            o, _, _ = self.dec(z * x_mask, nsff0, g=g)

        return o, x_mask, (z, z_p, m_p, logs_p)

