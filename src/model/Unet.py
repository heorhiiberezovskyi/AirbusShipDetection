import torch
import torchvision.transforms
from torch import nn, Tensor
from torch.nn import functional as F


class ImageNetNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_net_norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x / 255
        x = self.image_net_norm(x)
        return x


class ConvGNRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, gn_groups: int, relu_slope: float = 0.2):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=relu_slope)
        )

    def forward(self, x):
        return self.block(x)


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._residual = residual

        self.proc_1 = ConvGNRelu(in_channels, out_channels, gn_groups=8)
        self.proc_2 = ConvGNRelu(out_channels, out_channels, gn_groups=8)

    def forward(self, x):
        if self._residual:
            x = self.proc_1(x)
            y = self.proc_2(x)
            return x + y
        else:
            return self.proc_2(self.proc_1(x))


class DownConv(nn.Module):
    def __init__(self, channels: int, gn_groups: int):
        super().__init__()
        self._channels = channels

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(num_channels=channels, num_groups=gn_groups)
        )

    def forward(self, x):
        return self.block(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._channels = in_channels

        self.postprocess = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.postprocess(F.interpolate(x, scale_factor=2, mode="bilinear"))


class DecoderUpFuseModule(nn.Module):
    def __init__(self, in_channels: int, residual_block: bool):
        super().__init__()
        self._in_channels = in_channels
        self._residual_block = residual_block

        out_channels = in_channels // 2
        assert in_channels % 2 == 0

        self.up = UpConv(in_channels, out_channels)
        self.dec = Block(out_channels * 2, out_channels, residual_block)

    def forward(self, decoder_lower_channels: Tensor, encoder_channels: Tensor):
        result = self.up(decoder_lower_channels)
        result = torch.cat((result, encoder_channels), dim=1)
        result = self.dec(result)
        return result


class EncoderDownModule(nn.Module):
    def __init__(self, in_channels: int, residual_block: bool):
        super().__init__()
        self._in_channels = in_channels
        self._residual_block = residual_block

        self.down = DownConv(in_channels, gn_groups=8)
        self.enc = Block(in_channels, in_channels * 2, residual_block)

    def forward(self, x: Tensor):
        return self.enc(self.down(x))


class Unet(nn.Module):
    def __init__(self, init_channels: int, residual_block: bool):
        super().__init__()
        self._init_channels = init_channels
        self._residual_block = residual_block

        self.image_net_norm = ImageNetNorm()

        self.enc1 = Block(3, init_channels, residual_block)

        self.enc2 = EncoderDownModule(init_channels, residual_block)
        self.enc4 = EncoderDownModule(init_channels * 2, residual_block)
        self.enc8 = EncoderDownModule(init_channels * 4, residual_block)
        self.enc16 = EncoderDownModule(init_channels * 8, residual_block)

        self.dec8 = DecoderUpFuseModule(init_channels * 16, residual_block)
        self.dec4 = DecoderUpFuseModule(init_channels * 8, residual_block)
        self.dec2 = DecoderUpFuseModule(init_channels * 4, residual_block)
        self.dec1 = DecoderUpFuseModule(init_channels * 2, residual_block)

        self.out = nn.Conv2d(init_channels, 1, kernel_size=1)

    def forward(self, images: Tensor):
        images = self.image_net_norm(images)

        encoder_1 = self.enc1(images)
        encoder_2 = self.enc2(encoder_1)
        encoder_4 = self.enc4(encoder_2)
        encoder_8 = self.enc8(encoder_4)
        encoder_16 = self.enc16(encoder_8)

        decoder_8 = self.dec8(encoder_16, encoder_8)
        decoder_4 = self.dec4(decoder_8, encoder_4)
        decoder_2 = self.dec2(decoder_4, encoder_2)
        decoder_1 = self.dec1(decoder_2, encoder_1)

        out = self.out(decoder_1)

        out = torch.sigmoid(out)
        return out
