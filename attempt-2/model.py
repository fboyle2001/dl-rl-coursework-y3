import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F
import math
import op.up_or_down_sampling as up_or_down_sampling
from op.ddpm_init import default_init as ddpm_init
import string

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels. From Song et al repo"""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def activation_lookup(name):
    if name == "swish":
        return nn.SiLU()
    else:
        raise ValueError

def create_convolution_layer(kernel_size, in_ch, out_ch, init_scale=1):
    padding = 0 if kernel_size == 1 else 1
    conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, bias=True)

    # Apply DDPM Initial Scaling
    conv_layer.weight.data = ddpm_init(init_scale)(conv_layer.weight.data.shape)
    nn.init.zeros_(conv_layer.bias)

    return conv_layer

def create_linear_layer(in_ch, out_ch):
    linear_layer = nn.Linear(in_ch, out_ch)

    # Apply DDPM Initial Scaling
    linear_layer.weight.data = ddpm_init()(linear_layer.weight.shape)
    nn.init.zeros_(linear_layer.bias)

    return linear_layer

class BigGANResidualBlock(nn.Module):
    """
    Residual block, adapted from DDPM and NCSN++ layers
    """
    def __init__(self, in_ch, out_ch, direction=None, dropout=0.1, activation="swish", gn_eps=1e-6, time_embedding_dimension=None):
        super().__init__()
        assert time_embedding_dimension is not None, "Time Embedding Dimension cannot be None"

        self.activation_fn = activation_lookup(activation)

        self.group_norm_initial_layer = lambda y: self.activation_fn(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=gn_eps).cuda()(y))

        self.resampler = None
        self.final_batch_conv_layer = None

        if direction == "up":
            self.resampler = lambda y: up_or_down_sampling.upsample_2d(y, k=(1, 3, 3, 1))
        elif direction == "down":
            self.resampler = lambda y: up_or_down_sampling.downsample_2d(y, k=(1, 3, 3, 1))
        elif in_ch != out_ch:
            self.final_batch_conv_layer = create_convolution_layer(kernel_size=1, in_ch=in_ch, out_ch=out_ch)

        self.in_conv_3x3_layer = create_convolution_layer(kernel_size=3, in_ch=in_ch, out_ch=out_ch)
        self.skip_linear_layer = create_linear_layer(time_embedding_dimension, out_ch)
        self.group_norm_last_layer = lambda y: self.activation_fn(nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=gn_eps)(y))
        self.dropout_layer = nn.Dropout(dropout)
        self.out_conv_3x3_layer = create_convolution_layer(kernel_size=3, in_ch=out_ch, out_ch=out_ch, init_scale=0) # TODO: has different init scale to first conv?
        self.skip_rescale = lambda x, h: (x + h) / math.sqrt(2)

    def forward(self, batch, time_embedding):
        fwd_pass = self.group_norm_initial_layer(batch)

        if self.resampler is not None:
            fwd_pass = self.resampler(fwd_pass)
            batch = self.resampler(batch)

        fwd_pass = self.in_conv_3x3_layer(fwd_pass)
        fwd_pass += self.skip_linear_layer(time_embedding)[:, :, None, None]
        fwd_pass = self.dropout_layer(fwd_pass)
        fwd_pass = self.out_conv_3x3_layer(fwd_pass)

        if self.final_batch_conv_layer is not None:
            batch = self.final_batch_conv_layer(batch)

        return self.skip_rescale(batch, fwd_pass)

class MultiheadedSelfAttentionBlock(nn.Module):
    """
    V, K, Q, cite Attention is all you need paper
    """
    def __init__(self, in_ch):
        super().__init__()
        
        self.preparation_layer = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch)
        # num_heads = ?
        # print(f"emb {in_ch}, num_heads {8}")
        self.multihead_attention_layer = nn.MultiheadAttention(embed_dim=in_ch, num_heads=8, batch_first=True)

    def forward(self, batch):
        fwd_pass = self.preparation_layer(batch)

        # Need to check everything is working here since we are viewing and unviewing
        # might not be doing what I actually think it is 
        old_third = fwd_pass.shape[2]
        old_fourth = fwd_pass.shape[3]
        reshaped = fwd_pass.view(fwd_pass.shape[0], fwd_pass.shape[1], -1)
        reshaped = reshaped.permute(0, 2, 1)
        fwd_pass, _ = self.multihead_attention_layer(reshaped, reshaped, reshaped)
        fwd_pass = fwd_pass.permute(0, 2, 1)
        fwd_pass = fwd_pass.view(fwd_pass.shape[0], fwd_pass.shape[1], old_third, old_fourth)

        return (batch + fwd_pass) / math.sqrt(2)

class ProgressiveResidualSamplingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, direction=None):
        super().__init__()
        assert direction is not None, "Must set direction in PRSB"
        assert direction in ["up", "down"], "Direction in PRSB must be up or down"

        self.fir_convolution_layer = None

        # TODO: Add default init => kernel_init = ddpm_init()

        if direction == "up":
            self.fir_convolution_layer = up_or_down_sampling.Conv2d(in_ch, out_ch, kernel=3, up=True)
        else:
            self.fir_convolution_layer = up_or_down_sampling.Conv2d(in_ch, out_ch, kernel=3, down=True)

    def forward(self, batch):
        return self.fir_convolution_layer(batch)

class NCSNpp(nn.Module):
    def __init__(self, num_features, in_ch, device="cuda:0"):
        super().__init__()
        self.original_num_features = num_features
        self.original_in_ch = in_ch
        self.activiation = nn.SiLU()
        self.device = device

        self.fourier_embedding_layer = GaussianFourierProjection(embedding_size=num_features, scale=16)
        self.first_time_linear_layer = create_linear_layer(in_ch=num_features * 2, out_ch=num_features * 4)
        self.second_time_linear_layer = create_linear_layer(in_ch=num_features * 4, out_ch=num_features * 4)
        
        self.initial_convolution_layer = create_convolution_layer(kernel_size=3, in_ch=in_ch, out_ch=num_features)

        self.first_down_residual_block = self.create_biggan_combined_block(in_ch=num_features, out_ch=num_features, append=True)
        self.first_down_prs_block = self.create_progressive_sampling_block(res_in_ch=num_features, res_out_ch=num_features, prog_in_ch=in_ch, prog_out_ch=num_features, direction="down", append=True)
        
        self.second_down_residual_block = self.create_biggan_combined_block(in_ch=num_features, out_ch=num_features * 2, include_attn_blocks=True, append=True)
        self.second_down_prs_block = self.create_progressive_sampling_block(res_in_ch=num_features * 2, res_out_ch=num_features * 2, prog_in_ch=num_features, prog_out_ch=num_features * 2, direction="down", append=True)
        
        self.third_down_residual_block = self.create_biggan_combined_block(in_ch=num_features * 2, out_ch=num_features * 2, append=True)
        self.third_down_prs_block = self.create_progressive_sampling_block(res_in_ch=num_features * 2, res_out_ch=num_features * 2, prog_in_ch=num_features * 2, prog_out_ch=num_features * 2, direction="down", append=True)
        
        self.fourth_down_residual_block = self.create_biggan_combined_block(in_ch=num_features * 2, out_ch=num_features * 2, append=True)

        downsample_order = [
            # Downsampling
            self.first_down_residual_block,
            self.first_down_prs_block,
            self.second_down_residual_block,
            self.second_down_prs_block,
            self.third_down_residual_block,
            self.third_down_prs_block,
            self.fourth_down_residual_block
        ]
        
        self.downsample_order = downsample_order       
        self.intermediate_block = self.create_intermediate_block(ch=num_features * 2)

        self.first_up_single_res_block = self.create_biggan_combined_block(in_ch=num_features * 2, out_ch=num_features * 2, layer_count=1, direction="up")

        self.second_up_residual_block = self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=5)
        self.second_up_single_res_block = self.create_biggan_combined_block(in_ch=num_features * 2, out_ch=num_features * 2, layer_count=1, direction="up")

        self.third_up_residual_block_a = self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=4)
        self.third_up_residual_block_b = self.create_biggan_combined_block(in_ch=num_features * 3, out_ch=num_features * 2, layer_count=1)
        

        upsample_order = [
            # Upsampling
            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),
            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),
            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),
            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),

            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),
            (self.first_up_single_res_block, False),

            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),
            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),
            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),
            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),

            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),
            (self.second_up_single_res_block, False),

            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),
            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),
            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),
            (self.create_biggan_combined_block(in_ch=num_features * 4, out_ch=num_features * 2, layer_count=1), True),
            
            (self.create_biggan_combined_block(in_ch=num_features * 3, out_ch=num_features * 2, layer_count=1), True),
            (self.create_reverse_biggan_combined_block(in_ch=num_features * 2, out_ch=num_features * 2, direction="up"), False),

            (self.create_biggan_combined_block(in_ch=num_features * 3, out_ch=num_features, layer_count=1), True),
            (self.create_biggan_combined_block(in_ch=num_features * 2, out_ch=num_features, layer_count=1), True),
            (self.create_biggan_combined_block(in_ch=num_features * 2, out_ch=num_features, layer_count=1), True),
            (self.create_biggan_combined_block(in_ch=num_features * 2, out_ch=num_features, layer_count=1), True),
            (self.create_biggan_combined_block(in_ch=num_features * 2, out_ch=num_features, layer_count=1), True)
        ]

        self.upsample_order = upsample_order
        self.skip_connection_stack = []

        self.final_group_norm_layer = lambda y: self.activiation(nn.GroupNorm(num_groups=32, num_channels=num_features, eps=1e-6).to(self.device)(y))
        self.final_conv_layer = create_convolution_layer(kernel_size=3, in_ch=num_features, out_ch=in_ch, init_scale=0).to(self.device)

    def create_reverse_biggan_combined_block(self, in_ch, out_ch, direction=None):
        layers = [
            MultiheadedSelfAttentionBlock(in_ch=in_ch).to(self.device),
            BigGANResidualBlock(in_ch=in_ch, out_ch=out_ch, direction=direction, time_embedding_dimension=self.original_num_features * 4).to(self.device)
        ]

        def run(batch, input_skip, time_embedding):
            for i, layer in enumerate(layers):
                if i % 2 == 0:
                    batch = layer(batch)
                else:
                    batch = layer(batch, time_embedding)
            
            return batch, input_skip

        return run

    def create_biggan_combined_block(self, in_ch, out_ch, layer_count=4, include_attn_blocks=False, direction=False, append=False):
        layers = []
        # direction not setup on ATTN blocks

        if include_attn_blocks:
            layers = [
                BigGANResidualBlock(in_ch=in_ch, out_ch=out_ch, time_embedding_dimension=self.original_num_features * 4).to(self.device),
                MultiheadedSelfAttentionBlock(in_ch=out_ch).to(self.device)
            ]

            for i in range(layer_count - 1):
                layers.append(BigGANResidualBlock(in_ch=out_ch, out_ch=out_ch, time_embedding_dimension=self.original_num_features * 4).to(self.device))
                layers.append(MultiheadedSelfAttentionBlock(in_ch=out_ch).to(self.device))
        else:
            layers = [
                BigGANResidualBlock(in_ch=in_ch, out_ch=out_ch, time_embedding_dimension=self.original_num_features * 4, direction=direction).to(self.device),
            ]

            for i in range(layer_count - 1):
                layers.append(BigGANResidualBlock(in_ch=out_ch, out_ch=out_ch, time_embedding_dimension=self.original_num_features * 4, direction=direction).to(self.device))

        def run(batch, input_skip, time_embedding):
            for i, layer in enumerate(layers):
                if i % 2 == 0 or not include_attn_blocks:
                    batch = layer(batch, time_embedding)
                    if append:
                        self.skip_connection_stack.append(batch)
                else:
                    batch = layer(batch)

            return batch, input_skip

        return run

    def create_progressive_sampling_block(self, res_in_ch, res_out_ch, prog_in_ch, prog_out_ch, direction=None, append=False):
        assert direction in ["up", "down"], "Direction in PSB must be up or down"

        res_block = BigGANResidualBlock(in_ch=res_in_ch, out_ch=res_out_ch, direction=direction, time_embedding_dimension=self.original_num_features * 4).to(self.device)
        prg_block = ProgressiveResidualSamplingBlock(in_ch=prog_in_ch, out_ch=prog_out_ch, direction=direction).to(self.device)

        def run(batch, input_skip, time_embedding):
            batch = res_block(batch, time_embedding)
            input_skip = prg_block(input_skip)
            input_skip = (input_skip + batch) / math.sqrt(2)
            if append:
                self.skip_connection_stack.append(input_skip)
            # batch = input_skip
            return input_skip, input_skip

        return run

    def create_intermediate_block(self, ch):
        first_res_block = BigGANResidualBlock(in_ch=ch, out_ch=ch, time_embedding_dimension=self.original_num_features * 4).to(self.device)
        attn_block = MultiheadedSelfAttentionBlock(in_ch=ch).to(self.device)
        second_res_block = BigGANResidualBlock(in_ch=ch, out_ch=ch, time_embedding_dimension=self.original_num_features * 4).to(self.device)

        def run(batch, input_skip, time_embedding):
            fwd_pass = first_res_block(batch, time_embedding)
            fwd_pass = attn_block(batch)
            fwd_pass = second_res_block(batch, time_embedding)
            return fwd_pass, input_skip

        return run

    def forward(self, batch, t):
        time_embedding = self.fourier_embedding_layer(torch.log(t))
        time_embedding = self.first_time_linear_layer(time_embedding)
        time_embedding = self.second_time_linear_layer(self.activiation(time_embedding))

        batch = 2 * batch - 1 # L261 Song et al.
        input_skip = batch
        fwd_pass = self.initial_convolution_layer(batch)
        self.skip_connection_stack = [fwd_pass]
        
        # Downsampling
        for i, block in enumerate(self.downsample_order):
            fwd_pass, input_skip = block(fwd_pass, input_skip, time_embedding)
        
        # Intermediate
        fwd_pass, input_skip = self.intermediate_block(fwd_pass, input_skip, time_embedding)

        # Upsampling with skip connections
        for j, (block, should_pop) in enumerate(self.upsample_order):
            fwd_input = fwd_pass

            if should_pop:
                popped = self.skip_connection_stack.pop()
                fwd_input = torch.cat([fwd_input, popped], dim=1)

            fwd_pass, input_skip = block(fwd_input, input_skip, time_embedding)

        fwd_pass = self.final_group_norm_layer(fwd_pass)
        fwd_pass = self.final_conv_layer(fwd_pass)

        # print("T", t.shape)
        # Not entirely sure what this is doing here, some form of rescaling? on the channels? (batch_size, img_channels)?
        t_shape = t.reshape((batch.shape[0], *([1] * len(batch.shape[1:]))))
        # print("TS, FWP", t_shape.shape, fwd_pass.shape)
        fwd_pass = fwd_pass / t_shape
        return fwd_pass
        