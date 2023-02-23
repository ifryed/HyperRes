from asyncio import open_unix_connection
from multiprocessing.managers import BaseListProxy
from os import device_encoding
from models.HyperRes import HyperConv
from models.parallel_utils import ModuleParallel, convParallel
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperConvTranspose(nn.Module):
    def __init__(self,
                 levels,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 gruops=1,
                 bias=True,
                 padding_mode='zeros'
                 ):
        super().__init__()

        self.levels = levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = gruops
        self.bias = bias
        self.padding = padding
        self.padding_mode = padding_mode

        self.fc = nn.Linear(1, self.out_channels*(
            self.in_channels*(kernel_size**2) + int(self.bias)
        ))
        self.w_len = self.in_channels * \
            (self.out_channels * self.kernel_size**2)

    def forward(self, x):
        out = [None for _ in range(len(self.levels))]
        scale = [torch.tensor([l]).type(torch.float32).to(
            x[0].device) for l in self.levels]
        for i in range(len(scale)):
            tot_weights = self.fc(scale[i])
            weights = tot_weights[:self.w_len].resize(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
            bias = tot_weights[self.w_len:]
            bias = bias if self.bias else None
            out[i] = F.conv_transpose2d(x[i], weights, bias,
                                        stride=self.stride,
                                        padding=self.padding,
                                        dilation=self.dilation)
        return out


@torch.no_grad()
def init_weights(init_type='xavier'):
    if init_type == 'xavier':
        init = nn.init.xavier_normal
    elif init_type == 'he':
        init = nn.init.kaiming_normal
    else:
        init = nn.init.orthogonal

    def initializer(m):
        classname = m.__class__.__name__
        if classname.find('HyperConv') != -1:
            init(m.fc.weight)
            # m.fc.weight.data *= 1e-6
            m.fc.bias.data.zero_()
        elif classname.find('Conv2d') != -1:
            init(m.weight)
            m.bias.data.zero_()
    return initializer


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, levels) -> None:
        super().__init__()
        self.levels = levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device_encoding

        self.conv = HyperConv(self.levels, in_channels,
                              self.out_channels, kernel_size=2, stride=2)
        self.actv = ModuleParallel(nn.PReLU(self.out_channels))

    def forward(self, x):
        out = self.conv(x)
        return self.actv(out)
        # return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels, levels) -> None:
        super().__init__()

        self.levels = levels
        self.in_channels = in_channels
        self.cat_channels = cat_channels
        self.out_channels = out_channels

        self.conv = HyperConv(
            self.levels, in_channels+cat_channels, out_channels, kernel_size=3, padding=1)
        self.conv_t = HyperConvTranspose(levels, in_channels, in_channels, 2, stride=2)

        self.actv = ModuleParallel(nn.PReLU(self.out_channels))
        self.actv_t = ModuleParallel(nn.PReLU(self.in_channels))

    def forward(self, x):
        upsample, concat = x

        out = self.conv_t(upsample)
        upsample = self.actv_t(out)
        # upsample = out

        out = [torch.cat([con, ups], 1) for con, ups in zip(concat, upsample)]

        out = self.conv(out)
        return self.actv(out)
        # return out


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv_1 = convParallel(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = convParallel(out_channels, out_channels, kernel_size=3, padding=1)

        self.actv_1 = ModuleParallel(nn.PReLU(out_channels))
        self.actv_2 = ModuleParallel(nn.PReLU(out_channels))

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        x = self.actv_2(self.conv_2(x))

        return x


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv_1 = convParallel(in_channels, in_channels, 3, padding=1)
        self.conv_2 = convParallel(in_channels, out_channels, 3, padding=1)

        self.actv_1 = ModuleParallel(nn.PReLU(in_channels))
        self.actv_2 = ModuleParallel(nn.PReLU(out_channels))

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        x = self.actv_2(self.conv_2(x))

        return x


class DenoiseBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, levels):
        super().__init__()

        self.levels = [l/255 for l in levels]

        self.conv_0 = HyperConv(self.levels, 
                                in_channels, inner_channels, kernel_size=3, padding=1)
        self.conv_1 = HyperConv(self.levels,
                                in_channels+inner_channels, inner_channels, kernel_size=3, padding=1)
        self.conv_2 = HyperConv(self.levels,
                                in_channels + 2*inner_channels, inner_channels, kernel_size=3, padding=1)
        self.conv_3 = HyperConv(self.levels,
                                in_channels + 3*inner_channels, out_channels, kernel_size=3, padding=1)

        self.actv_0 = ModuleParallel(nn.PReLU(inner_channels))
        self.actv_1 = ModuleParallel(nn.PReLU(inner_channels))
        self.actv_2 = ModuleParallel(nn.PReLU(inner_channels))
        self.actv_3 = ModuleParallel(nn.PReLU(out_channels))

    def forward(self, x):
        out_0 = self.actv_0(self.conv_0(x))
        out_0 = [torch.cat([xx, oo],1) for xx, oo in zip(x, out_0)]

        out_1 = self.actv_1(self.conv_1(out_0))
        out_1 = [torch.cat([xx, oo],1) for xx, oo in zip(out_1, out_0)]

        out_2 = self.actv_2(self.conv_2(out_1))
        out_2 = [torch.cat([xx, oo],1) for xx, oo in zip(out_2, out_1)]

        out_3 = self.actv_3(self.conv_3(out_2))
        return [xx+oo for xx, oo in zip(out_3, x)]


class RDUNet(nn.Module):

    def __init__(self, levels, channels=3, filters=128) -> None:
        super().__init__()

        self.levels = levels

        filters_0 = filters
        filters_1 = 2 * filters_0
        filters_2 = 4 * filters_0
        filters_3 = 8 * filters_0

        # Levels 0
        self.input_block = InputBlock(channels, filters_0)
        self.block_0_0 = DenoiseBlock(
            filters_0, filters_0//2, filters_0, self.levels)
        self.block_0_1 = DenoiseBlock(
            filters_0, filters_0//2, filters_0, self.levels)
        self.down_0 = DownsampleBlock(
            filters_0, filters_1, self.levels)

        # Levels 1
        self.block_1_0 = DenoiseBlock(
            filters_1, filters_1//2, filters_1, self.levels)
        self.block_1_1 = DenoiseBlock(
            filters_1, filters_1//2, filters_1, self.levels)
        self.down_1 = DownsampleBlock(
            filters_1, filters_2, self.levels)

        # Levels 2
        self.block_2_0 = DenoiseBlock(
            filters_2, filters_2//2, filters_2, self.levels)
        self.block_2_1 = DenoiseBlock(
            filters_2, filters_2//2, filters_2, self.levels)
        self.down_2 = DownsampleBlock(
            filters_2, filters_3, self.levels)

        # Levels 3 (Bottle Neck)
        self.block_3_0 = DenoiseBlock(
            filters_3, filters_3//2, filters_3, self.levels)
        self.block_3_1 = DenoiseBlock(
            filters_3, filters_3//2, filters_3, self.levels)

        # Decoder
        # Level 2
        self.up_2 = UpsampleBlock(
            filters_3, filters_2, filters_2, self.levels)
        self.block_2_2 = DenoiseBlock(
            filters_2, filters_2//2, filters_2, self.levels)
        self.block_2_3 = DenoiseBlock(
            filters_2, filters_2//2, filters_2, self.levels)

        # Level 1
        self.up_1 = UpsampleBlock(
            filters_2, filters_1, filters_1, self.levels)
        self.block_1_2 = DenoiseBlock(
            filters_1, filters_1//2, filters_1, self.levels)
        self.block_1_3 = DenoiseBlock(
            filters_1, filters_1//2, filters_1, self.levels)

        # Level 0
        self.up_0 = UpsampleBlock(
            filters_1, filters_0, filters_0, self.levels)
        self.block_0_2 = DenoiseBlock(
            filters_0, filters_0//2, filters_0, self.levels)
        self.block_0_3 = DenoiseBlock(
            filters_0, filters_0//2, filters_0, self.levels)

        self.output_block = OutputBlock(filters_0, channels)
        self.sigmoid = ModuleParallel(nn.Sigmoid())

    def forward(self, inputs):
        out_0 = self.input_block(inputs)
        out_0 = self.block_0_0(out_0)
        out_0 = self.block_0_1(out_0)

        out_1 = self.down_0(out_0)
        out_1 = self.block_1_0(out_1)
        out_1 = self.block_1_1(out_1)

        out_2 = self.down_1(out_1)
        out_2 = self.block_2_0(out_2)
        out_2 = self.block_2_1(out_2)

        out_3 = self.down_2(out_2)
        out_3 = self.block_3_0(out_3)
        out_3 = self.block_3_1(out_3)

        out_4 = self.up_2([out_3, out_2])
        out_4 = self.block_2_2(out_4)
        out_4 = self.block_2_3(out_4)

        out_5 = self.up_1([out_4, out_1])
        out_5 = self.block_1_2(out_5)
        out_5 = self.block_1_3(out_5)

        out_6 = self.up_0([out_5, out_0])
        out_6 = self.block_0_2(out_6)
        out_6 = self.block_0_3(out_6)

        out_6 = self.output_block(out_6)
        out = [o_6 + inp for o_6, inp in zip(out_6, inputs)]

        return self.sigmoid(out)
