import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SegNetEnc(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class SegNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        decoders = list(models.vgg16(pretrained=True).features.children())

        self.dec1 = nn.Sequential(*decoders[:5])
        self.dec2 = nn.Sequential(*decoders[5:10])
        self.dec3 = nn.Sequential(*decoders[10:17])
        self.dec4 = nn.Sequential(*decoders[17:24])
        self.dec5 = nn.Sequential(*decoders[24:])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = False

        # Encoder stages for shared feature extraction
        self.enc5 = SegNetEnc(512, 512, 1)
        self.enc4 = SegNetEnc(1024, 256, 1)
        self.enc3 = SegNetEnc(512, 128, 1)
        self.enc2 = SegNetEnc(256, 64, 0)
        self.enc1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Separate output heads for SIC, SOD, and FLOE tasks
        self.final_sic = nn.Conv2d(64, n_classes['SIC'], 3, padding=1)
        self.final_sod = nn.Conv2d(64, n_classes['SOD'], 3, padding=1)
        self.final_floe = nn.Conv2d(64, n_classes['FLOE'], 3, padding=1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)
        enc5 = self.enc5(dec5)

        enc4 = self.enc4(torch.cat([dec4, enc5], 1))
        enc3 = self.enc3(torch.cat([dec3, enc4], 1))
        enc2 = self.enc2(torch.cat([dec2, enc3], 1))
        enc1 = self.enc1(torch.cat([dec1, enc2], 1))

        # Generate outputs for each task
        output_sic = F.interpolate(self.final_sic(enc1), x.size()[2:], mode='bilinear', align_corners=False)
        output_sod = F.interpolate(self.final_sod(enc1), x.size()[2:], mode='bilinear', align_corners=False)
        output_floe = F.interpolate(self.final_floe(enc1), x.size()[2:], mode='bilinear', align_corners=False)

        return output_sic, output_sod, output_floe
