import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F
import string

## Helper functions not entirely sure what they do
def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init

def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return torch.einsum(einsum_str, x, y)

def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)

def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

class NIN(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    y = contract_inner(x, self.W) + self.b
    return y.permute(0, 3, 1, 2)

## Define convolutions with weights and bias set

""" 
1x1 convolution
"""
def conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
  """1x1 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv

"""
3x3 convolution
"""
def conv3x3(in_ch, out_ch, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv

## Up and Downsampling
def resample2D(x, sample_direction, device=None):
    r"""
    Takes shape [N, C, H, W] and downsamples to [N, C, H // 2, W // 2]
    or [N, C, H, W] upsampled to [N, C, H * 2, W * 2]
    """

    if device is None:
        print("[WARN] No device specified, defaulting to cuda")
        device = "cuda:0"

    if sample_direction == None or sample_direction.lower() not in ["up", "down"]:
        return None

    up = sample_direction.lower() == "up"
    N, C, H, W = x.size()
    resample_layer = None

    if up:
        resample_layer = nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1, device=device)
    else:
        resample_layer = nn.Conv2d(C, C, kernel_size=4, stride=2, padding=1, device=device)
    
    resampled = resample_layer(x)
    return resampled

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

"""
Input is [N, in_ch, H, W],
Down output is [N, out_ch, H // 2, W // 2]
Up output is [N, out_ch, H * 2, W * 2]
"""
class ResNetBlock(nn.Module):
    def __init__(self, activation_fn, in_ch, out_ch, sample_direction, dropout, device):
        super().__init__()

        self.activation_fn = activation_fn
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.sample_direction = None if sample_direction is None else sample_direction.lower() 
        self.dropout = dropout
        self.device = device

        if sample_direction not in ["up", "down", None]:
            raise NotImplementedError(f"sample_direction must be up, down or {None}, received {sample_direction}")

        self.GNM0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        
        # Up/down sampling
        # self.RSP0 = 
        self.CNV0 = conv3x3(in_ch, out_ch)
        self.GNM1 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        self.DRP0 = nn.Dropout(dropout)
        self.CNV1 = conv3x3(out_ch, out_ch)

        # Up/down sampling or in_ch != out_ch
        self.CNV2 = conv1x1(in_ch, out_ch)

        # Finish with a rescale in forward

    def forward(self, x):
        h = self.activation_fn(self.GNM0(x))

        # Up/down sampling
        if self.sample_direction is not None:
            h = resample2D(h, self.sample_direction, device=self.device)
            x = resample2D(x, self.sample_direction, device=self.device)

        h = self.CNV0(h)
        h = self.activation_fn(self.GNM1(h))
        h = self.DRP0(h)
        h = self.CNV1(h)

        # Up/down sampling or in_ch != out_ch
        if self.sample_direction is not None or self.in_ch != self.out_ch:
            x = self.CNV2(x)

        return (x + h) / np.sqrt(2.)

"""
Need to read up on what this is, channel-wise self-attention block
Input is [N, channels, H, W]
Output is [N, channels, H, W]
"""
class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        self.GNM0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels)
        self.NIN0 = NIN(channels, channels)
        self.NIN1 = NIN(channels, channels)
        self.NIN2 = NIN(channels, channels)
        self.NIN3 = NIN(channels, channels, init_scale=0.)
    
    def forward(self, x):
        B, C, H, W = x.size()

        h = self.GNM0(x)
        q = self.NIN0(h)
        k = self.NIN1(h)
        v = self.NIN2(h)

        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN3(h)
        return x + h

"""
Input is [N, in_ch, H, W]
Output is [N, out_ch, H // 2, W // 2]
"""
class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.CNV0 = conv3x3(in_ch, out_ch, stride=2, padding=0)

    def forward(self, x):
        N, C, H, W = x.size()

        x = F.pad(x, (0, 1, 0, 1))
        x = self.CNV0(x)

        return x

if __name__ == "__main__":
    import torchvision
    import torchsummary

    def load_cifar10(img_width, batch_size=64):
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('../data', train=True, download=False, transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(img_width)
            ])),
        shuffle=True, batch_size=batch_size, drop_last=True)

        return train_loader

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader = load_cifar10(img_width=32, batch_size=64)

    in_ch = 64
    out_ch = 32

    # res_block = ResNetBlock(nn.SiLU(), in_ch, out_ch, sample_direction=None, dropout=0.1, device=device).to(device)
    # torchsummary.summary(res_block, (in_ch, 32, 32), batch_size=64)

    # attn_block = AttnBlock(in_ch).to(device)
    # torchsummary.summary(attn_block, (in_ch, 32, 32), batch_size=64)

    # down_block = DownsampleBlock(in_ch, out_ch).to(device)
    # torchsummary.summary(down_block, (in_ch, 32, 32), batch_size=64)


