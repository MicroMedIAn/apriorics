import torch
from torch import nn
from .convolution import ConvBnRelu
from torch.nn.functional import interpolate


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale**2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffleICNR(nn.Module):
    def __init__(self, in_chans, out_channels, bias=True, scale_factor=2, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans, out_channels * scale_factor**2, 1, bias=bias, **kwargs
        )
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale_factor)
        # self.pad = nn.ReflectionPad2d((1, 0, 1, 0))
        # self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.shuf(x)
        # x = self.pad(x)
        # x = self.blur(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_chans,
        skip_chans,
        hook,
        final_div=True,
        norm_layer=nn.BatchNorm2d,
        **kwargs
    ):
        super().__init__()
        self.hook = hook
        self.shuf = PixelShuffleICNR(in_chans, in_chans // 2, **kwargs)
        self.bn = norm_layer(skip_chans)
        ni = in_chans // 2 + skip_chans
        nf = ni if not final_div else skip_chans
        self.relu = nn.ReLU()
        self.conv1 = ConvBnRelu(ni, nf, 3, padding=1, norm_layer=norm_layer, **kwargs)
        self.conv2 = ConvBnRelu(nf, nf, 3, padding=1, norm_layer=norm_layer, **kwargs)

    def forward(self, x):
        skipco = self.hook.stored
        x = self.shuf(x)
        ssh = skipco.shape[-2:]
        if ssh != x.shape[-2:]:
            x = interpolate(x, ssh, mode="nearest")
        x = self.relu(torch.cat([x, self.bn(skipco)], dim=1))
        return self.conv2(self.conv1(x))
