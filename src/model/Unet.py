import torch
from torch import nn, Tensor

from src.model.Blocks import ImageNetNorm, DoubleConvBlock, EncoderDownModule, DecoderUpFuseModule


class Unet(nn.Module):
    def __init__(self, init_channels: int, residual_block: bool, inference: bool):
        super().__init__()
        self._inference = inference

        self._init_channels = init_channels
        self._residual_block = residual_block

        self.image_net_norm = ImageNetNorm()

        self.enc1 = DoubleConvBlock(3, init_channels, residual_block)

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

        if self._inference:
            out = torch.sigmoid(out)
        return out
