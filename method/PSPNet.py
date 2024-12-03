import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet101
from collections import OrderedDict

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class PyramidPooling(nn.Module):
    def __init__(self, name, out_planes, fc_dim=4096, pool_scales=[1, 2, 3, 6],
                 norm_layer=nn.BatchNorm2d):
        super(PyramidPooling, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(OrderedDict([
                ('{}/pool_1'.format(name), nn.AdaptiveAvgPool2d(scale)),
                ('{}/cbr'.format(name),
                 ConvBnRelu(fc_dim, 512, 1, 1, 0, has_bn=True,
                            has_relu=True, has_bias=False,
                            norm_layer=norm_layer))
            ])))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv6 = nn.Sequential(
            ConvBnRelu(fc_dim + len(pool_scales) * 512, 512, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False)
        )

    def forward(self, x):
        input_size = x.size()
        ppm_out = [x]
        for pooling in self.ppm:
            ppm_out.append(
                F.interpolate(pooling(x), size=(input_size[2], input_size[3]),
                              mode='bilinear', align_corners=True))
        ppm_out = torch.cat(ppm_out, 1)

        ppm_out = self.conv6(ppm_out)
        return ppm_out

class FeatureMap(nn.Module):
    """Class to perform final 1D convolution for each task."""
    def __init__(self, input_n, output_n):
        super(FeatureMap, self).__init__()
        self.feature_out = nn.Conv2d(input_n, output_n, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.feature_out(x)

class PSPNet(nn.Module):
    def __init__(self, options, bn_momentum=0.01):
        super(PSPNet, self).__init__()
        self.Resnet101 = resnet101.get_resnet101(dilation=[1, 1, 1, 2], bn_momentum=bn_momentum, is_fpn=False)
        self.psp_layer = PyramidPooling('psp', options['n_classes']['SIC'], 2048, norm_layer=nn.BatchNorm2d)

        # Multi-task output layers
        self.sic_feature_map = FeatureMap(512, options['n_classes']['SIC'])
        self.sod_feature_map = FeatureMap(512, options['n_classes']['SOD'])
        self.floe_feature_map = FeatureMap(512, options['n_classes']['FLOE'])

    def forward(self, input):
        b, c, h, w = input.shape
        x = self.Resnet101(input)
        psp_fm = self.psp_layer(x)

        # Resize feature maps for SIC, SOD, FLOE
        pred_sic = F.interpolate(self.sic_feature_map(psp_fm), size=input.size()[2:4], mode='bilinear', align_corners=True)
        pred_sod = F.interpolate(self.sod_feature_map(psp_fm), size=input.size()[2:4], mode='bilinear', align_corners=True)
        pred_floe = F.interpolate(self.floe_feature_map(psp_fm), size=input.size()[2:4], mode='bilinear', align_corners=True)

        return {'SIC': pred_sic, 'SOD': pred_sod, 'FLOE': pred_floe}

def main():
    options = {
        'n_classes': {'SIC': 10, 'SOD': 5, 'FLOE': 7}
    }
    in_batch, inchannel, in_h, in_w = 4, 3, 128, 128
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = PSPNet(options=options)
    out = net(x)
    print({k: v.shape for k, v in out.items()})

if __name__ == '__main__':
    main()
