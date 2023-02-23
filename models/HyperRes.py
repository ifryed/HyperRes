import torch
import torch.nn as nn
import torch.nn.functional as F
from .parallel_utils import ModuleParallel, convParallel
from torch.nn.common_types import _size_2_t


class HyperConv(nn.Module):
    def __init__(self,
                 levels,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 device='cpu'):
        super(HyperConv, self).__init__()

        self.levels = levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.device = device

        self.fc = nn.Linear(1, self.out_channels *
                            (self.in_channels * (kernel_size * kernel_size) + 1))
        self.w_len = self.in_channels * \
                     (self.out_channels * (self.kernel_size * self.kernel_size))

    def forward(self, x):
        out = [None for _ in range(len(self.levels))]
        scale = [torch.tensor([l]).type(torch.float32).to(x[0].device) for l in self.levels]

        for i in range(len(scale)):
            tot_weights = self.fc(scale[i])
            weights = tot_weights[:self.w_len].reshape(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
            bias = tot_weights[self.w_len:]
            out[i] = F.conv2d(x[i], weights, bias,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation
                              )

        return out


class ResBlockMeta(nn.Module):

    def __init__(self, levels, ker_n, inplanes, deivce='cpu'):
        super(ResBlockMeta, self).__init__()
        # meta network
        self.device = deivce
        self.levels = levels
        self.inplanes = inplanes
        self.ker_n = ker_n

        self.hy_conv1 = HyperConv(self.levels, ker_n, inplanes, kernel_size=3, stride=1, padding=1, device=self.device)
        self.hy_conv2 = HyperConv(self.levels, ker_n, ker_n, kernel_size=3, stride=1, padding=1, device=self.device)

        self.relu = ModuleParallel(nn.ReLU(inplace=True))

    def forward(self, x):
        identity = x
        x = self.hy_conv1(x)
        x = self.relu(x)
        out = self.hy_conv2(x)

        out = [out_i + in_i for out_i, in_i in zip(out, identity)]
        return out


class HyperRes(nn.Module):

    def __init__(self, meta_blocks, level=None, device='cpu', gray=False,
                 norm_factor=255):
        super(HyperRes, self).__init__()

        self.level = level
        if min(self.level) >= 1:
            self.level = [x / norm_factor for x in self.level]

        self.device = device
        self.inplanes = 64
        self.outplanes = 64
        self.dilation = 1
        self.num_parallel = len(self.level)

        self.channels = 1 if gray else 3
        self.conv1 = convParallel(self.channels, self.inplanes, kernel_size=3, stride=2, groups=1, padding=1, bias=True)

        self.res_blocks_meta = []
        self.res_blocks_meta.append(ModuleParallel(nn.Identity()))
        for _ in range(meta_blocks):
            self.res_blocks_meta.append(
                ResBlockMeta(self.level, self.inplanes, self.outplanes, deivce=self.device))
        self.res_blocks_meta = nn.Sequential(*self.res_blocks_meta)

        self.conv2 = convParallel(self.inplanes, self.inplanes, kernel_size=3, stride=1, groups=1, padding=1, bias=True)
        self.conv3 = convParallel(self.inplanes, self.inplanes * (2 ** 2), kernel_size=3, stride=1, groups=1, padding=1,
                                  bias=True)

        self.pix_shuff = ModuleParallel(nn.PixelShuffle(2))
        self.relu3 = ModuleParallel(nn.ReLU(inplace=True))

        self.conv4 = convParallel(self.inplanes, self.inplanes, kernel_size=3, stride=1, groups=1, padding=1, bias=True)
        self.relu4 = ModuleParallel(nn.ReLU(inplace=True))
        self.conv5 = convParallel(self.inplanes, self.channels, kernel_size=3, stride=1, groups=1, padding=1, bias=True)

    def forward(self, x):
        conv1_out = self.conv1(x)

        x = self.res_blocks_meta(conv1_out)

        x = self.conv2(x)
        x = [c_out + res_out for c_out, res_out in zip(conv1_out, x)]  # Skip connection
        x = self.conv3(x)
        x = self.pix_shuff(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        out = self.conv5(x)

        return out

    def setLevel(self, level):
        if not isinstance(level, list):
            level = [level]
        if max(level) >= 1:
            level = [x / 255 for x in level]

        self.level = level
        for k, v in self.res_blocks_meta._modules.items():
            for k, v2 in v._modules.items():
                if hasattr(v2, 'levels'):
                    v2.levels = level
