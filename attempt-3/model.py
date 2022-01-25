import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F
import math
import op.up_or_down_sampling as up_or_down_sampling
from op.ddpm_init import default_init as ddpm_init
from collections import namedtuple

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels. From Song et al repo"""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

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

        self.activation_fn = nn.SiLU()
        self.group_norm_initial_layer = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=gn_eps)

        self.direction = direction
        self.final_batch_conv_layer = None

        if in_ch != out_ch or direction is not None:
            self.final_batch_conv_layer = create_convolution_layer(kernel_size=1, in_ch=in_ch, out_ch=out_ch)

        self.in_conv_3x3_layer = create_convolution_layer(kernel_size=3, in_ch=in_ch, out_ch=out_ch)
        self.skip_linear_layer = create_linear_layer(time_embedding_dimension, out_ch)
        self.group_norm_last_layer = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=gn_eps)
        self.dropout_layer = nn.Dropout(dropout)
        self.out_conv_3x3_layer = create_convolution_layer(kernel_size=3, in_ch=out_ch, out_ch=out_ch, init_scale=0) # TODO: has different init scale to first conv?
        self.skip_rescale = lambda x, h: (x + h) / math.sqrt(2)

    def forward(self, batch, input_skip, time_embedding):
        fwd_pass = self.activation_fn(self.group_norm_initial_layer(batch))

        if self.direction == "up":
            fwd_pass = up_or_down_sampling.upsample_2d(fwd_pass, k=(1, 3, 3, 1))
            batch = up_or_down_sampling.upsample_2d(batch, k=(1, 3, 3, 1))
        elif self.direction == "down":
            fwd_pass = up_or_down_sampling.downsample_2d(fwd_pass, k=(1, 3, 3, 1))
            batch = up_or_down_sampling.downsample_2d(batch, k=(1, 3, 3, 1))

        fwd_pass = self.in_conv_3x3_layer(fwd_pass)
        fwd_pass += self.skip_linear_layer(time_embedding)[:, :, None, None]
        fwd_pass = self.activation_fn(self.group_norm_last_layer(fwd_pass))
        fwd_pass = self.dropout_layer(fwd_pass)
        fwd_pass = self.out_conv_3x3_layer(fwd_pass)

        if self.final_batch_conv_layer is not None:
            batch = self.final_batch_conv_layer(batch)

        return self.skip_rescale(batch, fwd_pass), input_skip

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

    def forward(self, batch, input_skip, time_embedding):
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

        return (batch + fwd_pass) / math.sqrt(2), input_skip

class ProgressiveResidualSamplingBlock(nn.Module):
    def __init__(self, res_in_ch, res_out_ch, fir_in_ch, fir_out_ch, time_embedding_dimension, direction):
        super().__init__()
        assert direction is not None, "Must set direction in PRSB"
        assert direction in ["up", "down"], "Direction in PRSB must be up or down"

        self.res_block = BigGANResidualBlock(in_ch=res_in_ch, out_ch=res_out_ch, direction=direction, time_embedding_dimension=time_embedding_dimension)
        self.fir_convolution_layer = None

        # TODO: Add default init => kernel_init = ddpm_init()

        if direction == "up":
            self.fir_convolution_layer = up_or_down_sampling.Conv2d(fir_in_ch, fir_out_ch, kernel=3, up=True)
        else:
            self.fir_convolution_layer = up_or_down_sampling.Conv2d(fir_in_ch, fir_out_ch, kernel=3, down=True)

    def forward(self, batch, input_skip, time_embedding):
        batch, input_skip = self.res_block(batch, input_skip, time_embedding)
        input_skip = self.fir_convolution_layer(input_skip)
        input_skip = (batch + input_skip) / math.sqrt(2)

        # batch = input_skip for output
        return input_skip, input_skip

class NCSNpp(nn.Module):
    def __init__(self, num_features, in_ch, device="cuda:0"):
        super().__init__()
        self.original_num_features = num_features
        self.original_in_ch = in_ch
        self.activation = nn.SiLU()
        self.device = device

        # The first 3 layers here are used to handle the time sigmas from the SDE
        # since the model being trained is time-dependant
        self.fourier_embedding_layer = GaussianFourierProjection(embedding_size=num_features, scale=16)
        self.first_time_linear_layer = create_linear_layer(in_ch=num_features * 2, out_ch=num_features * 4)
        self.second_time_linear_layer = create_linear_layer(in_ch=num_features * 4, out_ch=num_features * 4)
        self.initial_convolution_layer = create_convolution_layer(kernel_size=3, in_ch=in_ch, out_ch=num_features)

        # Used to determine whether to save the output or retrieve output for skip connections 
        ModuleMetadata = namedtuple("ModuleMetadata", ["stack_push", "stack_pop"])

        self.downsample_order = nn.ModuleList([
            # First residual block
            BigGANResidualBlock(in_ch=num_features, out_ch=num_features, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features, out_ch=num_features, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features, out_ch=num_features, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features, out_ch=num_features, time_embedding_dimension=self.original_num_features * 4),

            # First downsample block
            ProgressiveResidualSamplingBlock(res_in_ch=num_features, res_out_ch=num_features, fir_in_ch=in_ch, fir_out_ch=num_features, direction="down", time_embedding_dimension=self.original_num_features * 4),

            # Second residual block with attention
            BigGANResidualBlock(in_ch=num_features, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            MultiheadedSelfAttentionBlock(in_ch=num_features * 2),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            MultiheadedSelfAttentionBlock(in_ch=num_features * 2),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            MultiheadedSelfAttentionBlock(in_ch=num_features * 2),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            MultiheadedSelfAttentionBlock(in_ch=num_features * 2),

            # Second downsample block
            ProgressiveResidualSamplingBlock(res_in_ch=num_features * 2, res_out_ch=num_features * 2, fir_in_ch=num_features, fir_out_ch=num_features * 2, direction="down", time_embedding_dimension=self.original_num_features * 4),

            # Third residual block
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),

            # Third downsample block
            ProgressiveResidualSamplingBlock(res_in_ch=num_features * 2, res_out_ch=num_features * 2, fir_in_ch=num_features * 2, fir_out_ch=num_features * 2, direction="down", time_embedding_dimension=self.original_num_features * 4),
        
            # Fourth residual block
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
        ])

        # These are split rather than a list of tuples including the module since we need to register the modules with PyTorch for snapshotting
        self.downsample_metadata = [
            # First residual block
            ModuleMetadata(True, False),
            ModuleMetadata(True, False),
            ModuleMetadata(True, False),
            ModuleMetadata(True, False),

            # First downsample block
            ModuleMetadata(True, False),

            # Second residual block with attention
            ModuleMetadata(True, False),
            ModuleMetadata(False, False),
            ModuleMetadata(True, False),
            ModuleMetadata(False, False),
            ModuleMetadata(True, False),
            ModuleMetadata(False, False),
            ModuleMetadata(True, False),
            ModuleMetadata(False, False),

            # Second downsample block
            ModuleMetadata(True, False),

            # Third residual block
            ModuleMetadata(True, False),
            ModuleMetadata(True, False),
            ModuleMetadata(True, False),
            ModuleMetadata(True, False),

            # Third downsample block
            ModuleMetadata(True, False),

            # Fourth residual block
            ModuleMetadata(True, False),
            ModuleMetadata(True, False),
            ModuleMetadata(True, False),
            ModuleMetadata(True, False),
        ]

        self.intermediate_order = nn.ModuleList([
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            MultiheadedSelfAttentionBlock(in_ch=num_features * 2),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4)
        ])

        self.upsample_order = nn.ModuleList([
            # First residual block
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),

            # First upsample block
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4, direction="up"),
            
            # Second residual block
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),

            # Second upsample block
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4, direction="up"),

            # Third residual block
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 4, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),

            # Third upsample block
            BigGANResidualBlock(in_ch=num_features * 3, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4),
            MultiheadedSelfAttentionBlock(in_ch=num_features * 2),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features * 2, time_embedding_dimension=self.original_num_features * 4, direction="up"),

            # Fourth residual block
            BigGANResidualBlock(in_ch=num_features * 3, out_ch=num_features, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features, time_embedding_dimension=self.original_num_features * 4),
            BigGANResidualBlock(in_ch=num_features * 2, out_ch=num_features, time_embedding_dimension=self.original_num_features * 4),
        ])

        self.upsample_metadata = [
            # First residual block
            ModuleMetadata(False, True),
            ModuleMetadata(False, True),
            ModuleMetadata(False, True),
            ModuleMetadata(False, True),

            # First upsample block
            ModuleMetadata(False, True),
            ModuleMetadata(False, False),

            # Second residual block
            ModuleMetadata(False, True),
            ModuleMetadata(False, True),
            ModuleMetadata(False, True),
            ModuleMetadata(False, True),

            # Second upsample block
            ModuleMetadata(False, True),
            ModuleMetadata(False, False),

            # Third residual block
            ModuleMetadata(False, True),
            ModuleMetadata(False, True),
            ModuleMetadata(False, True),
            ModuleMetadata(False, True),

            # Third upsample block 
            ModuleMetadata(False, True),
            ModuleMetadata(False, False),
            ModuleMetadata(False, False),

            # Fourth residual block
            ModuleMetadata(False, True),
            ModuleMetadata(False, True),
            ModuleMetadata(False, True),
            ModuleMetadata(False, True),
            ModuleMetadata(False, True),
        ]

        self.final_group_norm_layer = nn.GroupNorm(num_groups=32, num_channels=num_features, eps=1e-6)
        self.final_conv_layer = create_convolution_layer(kernel_size=3, in_ch=num_features, out_ch=in_ch, init_scale=0)

    def forward(self, batch, t):
        time_embedding = self.fourier_embedding_layer(torch.log(t))
        time_embedding = self.first_time_linear_layer(time_embedding)
        time_embedding = self.second_time_linear_layer(self.activation(time_embedding))

        # Scaling as suggested by the original paper
        batch = 2 * batch - 1
        input_skip = batch
        fwd_pass = self.initial_convolution_layer(batch)
        fwd_pass_stack = [fwd_pass]

        # Forward pass on the main downsampling and upsampling blocks

        for block, metadata in zip(self.downsample_order, self.downsample_metadata):
            fwd_pass, input_skip = block(fwd_pass, input_skip, time_embedding)

            # Used for the skip connections in the upsampling block
            if metadata.stack_push:
                fwd_pass_stack.append(fwd_pass)

        for block in self.intermediate_order:
            fwd_pass, input_skip = block(fwd_pass, input_skip, time_embedding)

        for block, metadata in zip(self.upsample_order, self.upsample_metadata):
            upsample_input = fwd_pass

            # Get the inputs for the skip connections
            if metadata.stack_pop:
                popped = fwd_pass_stack.pop()
                # Concat on a feature level
                upsample_input = torch.cat([upsample_input, popped], dim=1)

            fwd_pass, input_skip = block(upsample_input, input_skip, time_embedding)

        fwd_pass = self.activation(self.final_group_norm_layer(fwd_pass))
        fwd_pass = self.final_conv_layer(fwd_pass)
        
        # Scale according to the time sigmas
        t_shape = t.reshape((batch.shape[0], *([1] * len(batch.shape[1:]))))
        fwd_pass = fwd_pass / t_shape

        return fwd_pass
        