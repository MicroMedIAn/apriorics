from torch import nn
import torch


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ConvBnRelu(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        eps: float = 1e-5,
        norm_layer: nn.Module = nn.BatchNorm2d,
        **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans,
            out_channels,
            kernel_size,
            stride=(stride, stride),
            padding=padding,
            bias=bias,
            **kwargs
        )
        self.bn = norm_layer(out_channels, eps=eps)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBn(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        eps: float = 1e-5,
        momentum: float = 0.01,
        **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans,
            out_channels,
            kernel_size,
            stride=(stride, stride),
            padding=padding,
            bias=bias,
            **kwargs
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvRelu(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans,
            out_channels,
            kernel_size,
            stride=(stride, stride),
            padding=padding,
            bias=bias,
            **kwargs
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        return x


class SelfAttentionBlock(nn.Module):
    """"""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int,
        stride: int = 1,
        groups: int = 8,
        bias: bool = False,
    ):
        super().__init__()
        assert (
            c_in % groups == c_out % groups == 0
        ), "c_in and c_out must be divided by groups"
        assert k % 2 == 1, "k must be odd"
        assert c_out % 2 == 0, "c_out must be even"

        padding = k // 2
        self.c_in = c_in
        self.c_out = c_out
        self.k = k
        self.stride = stride
        self.groups = groups

        self.key_conv = nn.Conv2d(
            c_in,
            c_out,
            1,
            padding=padding,
            groups=groups,
            bias=bias,
            padding_mode="reflect",
        )
        self.query_conv = nn.Conv2d(c_in, c_out, 1, groups=groups, bias=bias)
        self.value_conv = nn.Conv2d(
            c_in,
            c_out,
            1,
            padding=padding,
            groups=groups,
            bias=bias,
            padding_mode="reflect",
        )

        self.r_ai = nn.Parameter(torch.randn(1, c_out // 2, k, 1))
        self.r_aj = nn.Parameter(torch.randn(1, c_out // 2, 1, k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = self.c_out // self.groups

        k = (
            self.key_conv(x)
            .unfold(2, self.k, self.stride)
            .unfold(3, self.k, self.stride)
            .contiguous()
            .view(b, self.groups, n, h, w, -1)
        )
        r = torch.cat(
            (self.r_ai.expand(b, -1, -1, self.k), self.r_aj.expand(b, -1, self.k, -1)),
            dim=1,
        ).view(b, self.groups, n, -1)
        r = r[..., None, None, :].expand(-1, -1, -1, h, w, -1)

        out = k + r
        q = self.query_conv(x).view(b, self.groups, n, h, w, 1)
        out = q * out
        v = (
            self.value_conv(x)
            .unfold(2, self.k, self.stride)
            .unfold(3, self.k, self.stride)
            .contiguous()
            .view(b, self.groups, n, h, w, -1)
        )

        y = (
            (torch.softmax(out.sum(2, keepdims=True), dim=-1) * v)
            .sum(-1)
            .view(b, self.c_out, h, w)
        )

        return y


class LastCross(nn.Module):
    def __init__(
        self, n_chans: int, bottle: bool = False, norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__()
        n_mid = n_chans // 2 if bottle else n_chans
        self.conv1 = ConvBnRelu(n_chans, n_mid, 3, padding=1, norm_layer=norm_layer)
        self.conv2 = ConvBnRelu(n_mid, n_chans, 3, padding=1, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y
