import torch.nn as nn


def convParallel(in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1, padding=1, bias=True):
    """Parallel Conv layer"""
    return ModuleParallel(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, groups=groups, bias=bias, dilation=dilation)
    )


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class BatchNorm2dParallelAdv(nn.Module):
    def __init__(self, num_features, sigmas):
        super(BatchNorm2dParallelAdv, self).__init__()
        self.sigmas = sigmas
        for _, sig in enumerate(self.sigmas):
            setattr(self, "bn_" + str(sig), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, "bn_" + str(sig))(x) for sig, x in zip(self.sigmas, x_parallel)]
