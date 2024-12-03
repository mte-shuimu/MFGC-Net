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
                 norm_layer=nn.BatchNorm2d,
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
        self.map_bn = norm_layer(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_layer(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_layer(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)
        out = self.red_conv(out)

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

class Head(nn.Module):
    def __init__(self, classify_classes, norm_layer=nn.BatchNorm2d, bn_momentum=0.0003):
        super(Head, self).__init__()

        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [6, 12, 18], norm_layer=norm_layer)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            norm_layer(48, momentum=bn_momentum),
            nn.ReLU(),
            )

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )

        # 多任务输出层
        self.sic_classify = nn.Conv2d(256, classify_classes['SIC'], kernel_size=1, stride=1, padding=0, bias=True)
        self.sod_classify = nn.Conv2d(256, classify_classes['SOD'], kernel_size=1, stride=1, padding=0, bias=True)
        self.floe_classify = nn.Conv2d(256, classify_classes['FLOE'], kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, f_list):
        f = f_list[-1]
        f = self.aspp(f)

        low_level_features = f_list[0]
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)

        f = F.interpolate(f, size=(low_h, low_w), mode='bilinear', align_corners=True)
        f = torch.cat((f, low_level_features), dim=1)
        f = self.last_conv(f)

        # 各个任务的输出
        sic_pred = self.sic_classify(f)
        sod_pred = self.sod_classify(f)
        floe_pred = self.floe_classify(f)

        return sic_pred, sod_pred, floe_pred

class DeepLabV3plusMultiTask(nn.Module):
    def __init__(self, classify_classes, bn_momentum=0.01):
        super(DeepLabV3plusMultiTask, self).__init__()
        self.Resnet101 = resnet101.get_resnet101(dilation=[1, 1, 1, 2], bn_momentum=bn_momentum, is_fpn=True)
        self.head = Head(classify_classes, norm_layer=nn.BatchNorm2d)

    def forward(self, input):
        x = self.Resnet101(input)

        sic_pred, sod_pred, floe_pred = self.head(x)

        sic_output = F.interpolate(sic_pred, size=input.size()[2:4], mode='bilinear', align_corners=True)
        sod_output = F.interpolate(sod_pred, size=input.size()[2:4], mode='bilinear', align_corners=True)
        floe_output = F.interpolate(floe_pred, size=input.size()[2:4], mode='bilinear', align_corners=True)

        return sic_output, sod_output, floe_output

def main():
    classify_classes = {'SIC': 5, 'SOD': 2, 'FLOE': 3}
    in_batch, inchannel, in_h, in_w = 4, 3, 128, 128
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = DeepLabV3plusMultiTask(classify_classes=classify_classes)
    sic_out, sod_out, floe_out = net(x)
    print(sic_out.shape, sod_out.shape, floe_out.shape)

if __name__ == '__main__':
    main()
