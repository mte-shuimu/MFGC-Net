import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet101

class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(self.pooling_size[0], x.shape[2]),
                            min(self.pooling_size[1], x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool

class FeatureMap(nn.Module):
    """Class to perform final 1D convolution for each task."""
    def __init__(self, input_n, output_n):
        super(FeatureMap, self).__init__()
        self.feature_out = nn.Conv2d(input_n, output_n, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.feature_out(x)

class DeepLabV3(nn.Module):
    def __init__(self, options, bn_momentum=0.01):
        super(DeepLabV3, self).__init__()
        self.Resnet101 = resnet101.get_resnet101(dilation=[1, 1, 1, 2], bn_momentum=bn_momentum, is_fpn=False)
        self.ASPP = ASPP(2048, 256, [6, 12, 18], norm_act=nn.BatchNorm2d)

        # Multi-task output layers
        self.sic_feature_map = FeatureMap(256, options['n_classes']['SIC'])
        self.sod_feature_map = FeatureMap(256, options['n_classes']['SOD'])
        self.floe_feature_map = FeatureMap(256, options['n_classes']['FLOE'])

    def forward(self, input):
        x = self.Resnet101(input)
        aspp = self.ASPP(x)

        # Predictions for each task
        pred_sic = F.interpolate(self.sic_feature_map(aspp), size=input.size()[2:4], mode='bilinear', align_corners=True)
        pred_sod = F.interpolate(self.sod_feature_map(aspp), size=input.size()[2:4], mode='bilinear', align_corners=True)
        pred_floe = F.interpolate(self.floe_feature_map(aspp), size=input.size()[2:4], mode='bilinear', align_corners=True)

        return {'SIC': pred_sic, 'SOD': pred_sod, 'FLOE': pred_floe}

def main():
    options = {
        'n_classes': {'SIC': 10, 'SOD': 5, 'FLOE': 7}
    }
    in_batch, inchannel, in_h, in_w = 4, 3, 128, 128
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = DeepLabV3(options=options)
    out = net(x)
    print({k: v.shape for k, v in out.items()})

if __name__ == '__main__':
    main()
