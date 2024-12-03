import torch
import torch.nn as nn
import torch.nn.functional as F


class PoolFormerUNet(nn.Module):
    def __init__(self, poolformer, num_classes):
        super(PoolFormerUNet, self).__init__()

        self.encoder = poolformer
        self.num_classes_sic = num_classes['SIC']
        self.num_classes_sod = num_classes['SOD']
        self.num_classes_floe = num_classes['FLOE']

        # Decoder layers
        self.up_conv4 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.up_conv1 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)

        self.conv_block4 = self._conv_block(768 + 384, 384)
        self.conv_block3 = self._conv_block(384 + 192, 192)
        self.conv_block2 = self._conv_block(192 + 96, 96)
        self.conv_block1 = self._conv_block(96 + 48, 48)

        # Output heads for each task
        self.out_sic = nn.Conv2d(48, num_classes['SIC'], kernel_size=1)
        self.out_sod = nn.Conv2d(48, num_classes['SOD'], kernel_size=1)
        self.out_floe = nn.Conv2d(48, num_classes['FLOE'], kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        features = self.encoder.forward_tokens(x)

        # Decoder with skip connections
        x4, x3, x2, x1 = features[::-1]

        d4 = self.up_conv4(x4)
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.conv_block4(d4)

        d3 = self.up_conv3(d4)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.conv_block3(d3)

        d2 = self.up_conv2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.conv_block2(d2)

        d1 = self.up_conv1(d2)
        d1 = self.conv_block1(d1)

        # Outputs for each task
        out_sic = self.out_sic(d1)
        out_sod = self.out_sod(d1)
        out_floe = self.out_floe(d1)

        return out_sic, out_sod, out_floe
