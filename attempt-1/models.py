import torch
import torch.nn as nn
import numpy as np
from defined_blocks import ResNetBlock, AttnBlock, DownsampleBlock, GaussianFourierProjection, conv3x3

class NCSNpp(nn.Module):
    def __init__(self, in_ch, nf, activation_fn, device):
        super().__init__()

        self.in_ch = in_ch
        self.nf = nf
        self.activation_fn = activation_fn

        # all_res = [32, 16, 8, 4]

        self.num_res_blocks = num_res_blocks = 4
        self.ch_mult = ch_mult = [1, 2, 2, 2]
        self.attn_resolutions = attn_resolutions = [16]
        self.dropout = dropout = 0.1
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [32 // (2 ** i) for i in range(num_resolutions)]

        layers = [
            GaussianFourierProjection(embedding_size=nf, scale=16), # This is used for time_cond instead
            conv3x3(in_ch, nf)
        ]

        hs_c = [nf]
        in_ch = nf
        out_ch = 0
        input_pyramid_ch = self.in_ch

        for level in range(num_resolutions):
            for block in range(num_res_blocks):
                out_ch = nf * ch_mult[level]
                layers.append(ResNetBlock(activation_fn, in_ch, out_ch, sample_direction=None, dropout=dropout, device=device))
                in_ch = out_ch

                if all_resolutions[level] in attn_resolutions:
                    layers.append(AttnBlock(channels=in_ch))
                
                hs_c.append(in_ch)

            if level != num_resolutions - 1:
                layers.append(ResNetBlock(activation_fn, in_ch, out_ch=in_ch, sample_direction="down", dropout=dropout, device=device))
                layers.append(DownsampleBlock(in_ch=input_pyramid_ch, out_ch=in_ch))
                input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        layers.append(ResNetBlock(activation_fn, in_ch, out_ch=in_ch, sample_direction=None, dropout=dropout, device=device))
        layers.append(AttnBlock(channels=in_ch))
        layers.append(ResNetBlock(activation_fn, in_ch, out_ch=in_ch, sample_direction=None, dropout=dropout, device=device))

        pyramid_ch = 0

        for level in reversed(range(num_resolutions)):
            for block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[level]
                layers.append(ResNetBlock(activation_fn, in_ch=in_ch + hs_c.pop(), out_ch=out_ch, sample_direction=None, dropout=dropout, device=device))
                in_ch = out_ch

            if all_resolutions[level] in attn_resolutions:
                layers.append(AttnBlock(channels=in_ch))

            if level != 0:
                layers.append(ResNetBlock(activation_fn, in_ch=in_ch, out_ch=in_ch, sample_direction="up", dropout=dropout, device=device))

        assert not hs_c

        layers.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
        layers.append(conv3x3(in_ch, self.in_ch, init_scale=0.))

        self.layers = nn.ModuleList(layers)

    def forward(self, x, t):
        m_idx = 0
        modules = self.layers

        used_sigmas = t
        temb = modules[m_idx](torch.log(used_sigmas))
        m_idx += 1
        
        input_pyramid = x
        hs = [modules[m_idx](x)]
        m_idx += 1

        for level in range(self.num_resolutions):
            for block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1])
                m_idx += 1

                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                
                hs.append(h)
            
            if level != self.num_resolutions - 1:
                h = modules[m_idx](hs[-1])
                m_idx += 1
                input_pyramid = modules[m_idx](input_pyramid)
                m_idx += 1
                input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                h = input_pyramid
                
                hs.append(h)
        
        h = hs[-1]
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        for level in reversed(range(self.num_resolutions)):
            for block in range(self.num_res_blocks + 1):
                p = hs.pop()
                h = modules[m_idx](torch.cat([h, p], dim=1))
                m_idx += 1
            
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if level != 0:
                h = modules[m_idx](h)
                m_idx += 1
        
        assert not hs

        h = self.activation_fn(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(modules)

        used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
        h = h / used_sigmas
        
        return h



