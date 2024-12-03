import json
import math
import torch.nn as nn
import torch
import bricks
import torch.nn.functional as F
from timm.models import register_model


class StemConv(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=None):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            bricks.DownSampling(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                norm_layer=norm_layer
            ),
            bricks.DownSampling(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                norm_layer=norm_layer
            ),
        )

    def forward(self, x):
        out = self.proj(x)
        return out


class MSCA(nn.Module):

    def __init__(self, in_channels):
        super(MSCA, self).__init__()

        self.conv = bricks.DepthwiseConv(
            in_channels=in_channels,
            kernel_size=(5, 5),
            padding=(2, 2),
            bias=True
        )

        self.conv7 = nn.Sequential(
            bricks.DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                bias=True
            ),
            bricks.DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                bias=True
            )
        )

        self.conv11 = nn.Sequential(
            bricks.DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 11),
                padding=(0, 5),
                bias=True
            ),
            bricks.DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(11, 1),
                padding=(5, 0),
                bias=True
            )
        )

        self.conv21 = nn.Sequential(
            bricks.DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 21),
                padding=(0, 10),
                bias=True
            ),
            bricks.DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(21, 1),
                padding=(10, 0),
                bias=True
            )
        )

        self.fc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        u = x
        out = self.conv(x)

        branch1 = self.conv7(out)
        branch2 = self.conv11(out)
        branch3 = self.conv21(out)

        out = self.fc(out + branch1 + branch2 + branch3)
        out = out * u
        return out


class Attention(nn.Module):

    def __init__(self, in_channels):
        super(Attention, self).__init__()

        self.fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )
        self.msca = MSCA(in_channels=in_channels)
        self.fc2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        out = F.gelu(self.fc1(x))
        out = self.msca(out)
        out = self.fc2(out)
        return out


class FFN(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, drop_prob=0.):
        super(FFN, self).__init__()

        self.fc1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=(1, 1)
        )
        self.dw = bricks.DepthwiseConv(
            in_channels=hidden_features,
            kernel_size=(3, 3),
            bias=True
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=out_features,
            kernel_size=(1, 1)
        )
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        out = self.fc1(x)
        out = F.gelu(self.dw(out))
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):

    def __init__(self, in_channels, expand_ratio, drop_prob=0., drop_path_prob=0.):
        super(Block, self).__init__()

        self.norm1 = nn.BatchNorm2d(num_features=in_channels)
        self.attention = Attention(in_channels=in_channels)
        self.drop_path = bricks.DropPath(drop_prob=drop_path_prob if drop_path_prob >= 0 else nn.Identity)
        self.norm2 = nn.BatchNorm2d(num_features=in_channels)
        self.ffn = FFN(
            in_features=in_channels,
            hidden_features=int(expand_ratio * in_channels),
            out_features=in_channels,
            drop_prob=drop_prob
        )

        layer_scale_init_value = 1e-2
        self.layer_scale1 = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channels),
            requires_grad=True
        )
        self.layer_scale2 = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channels),
            requires_grad=True
        )

    def forward(self, x):
        out = self.norm1(x)
        out = self.attention(out)
        out = x + self.drop_path(
            self.layer_scale1.unsqueeze(-1).unsqueeze(-1) * out
        )
        x = out

        out = self.norm2(out)
        out = self.ffn(out)
        out = x + self.drop_path(
            self.layer_scale2.unsqueeze(-1).unsqueeze(-1) * out
        )

        return out


class Stage(nn.Module):

    def __init__(
            self,
            stage_id,
            in_channels,
            out_channels,
            expand_ratio,
            blocks_num,
            drop_prob=0.,
            drop_path_prob=[0.]
    ):
        super(Stage, self).__init__()

        assert blocks_num == len(drop_path_prob)

        if stage_id == 0:
            self.down_sampling = StemConv(
                in_channels=in_channels,
                out_channels=out_channels
            )
        else:
            self.down_sampling = bricks.DownSampling(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2)
            )

        self.blocks = nn.Sequential(
            *[
                Block(
                    in_channels=out_channels,
                    expand_ratio=expand_ratio,
                    drop_prob=drop_prob,
                    drop_path_prob=drop_path_prob[i]
                ) for i in range(0, blocks_num)
            ]
        )

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        out = self.down_sampling(x)
        out = self.blocks(out)
        batch_size, channels, height, width = out.shape
        out = out.view(batch_size, channels, -1)
        out = torch.transpose(out, -2, -1)
        out = self.norm(out)
        out = torch.transpose(out, -2, -1)
        out = out.view(batch_size, -1, height, width)

        return out


class MSCAN(nn.Module):

    def __init__(
            self,
            embed_dims=[3, 32, 64, 160, 256],
            expand_ratios=[8, 8, 4, 4],
            depths=[3, 3, 5, 2],
            drop_prob=0.1,
            drop_path_prob=0.1
    ):
        super(MSCAN, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_prob, sum(depths))]
        self.stages = nn.Sequential(
            *[
                Stage(
                    stage_id=stage_id,
                    in_channels=embed_dims[stage_id],
                    out_channels=embed_dims[stage_id + 1],
                    expand_ratio=expand_ratios[stage_id],
                    blocks_num=depths[stage_id],
                    drop_prob=drop_prob,
                    drop_path_prob=dpr[sum(depths[: stage_id]): sum(depths[: stage_id + 1])]
                ) for stage_id in range(0, len(depths))
            ]
        )

    def forward(self, x):
        out = x
        outputs = []

        for idx, stage in enumerate(self.stages):
            out = stage(out)
            if idx != 0:
                outputs.append(out)
        return [x, *outputs]


class Hamburger(nn.Module):

    def __init__(
            self,
            hamburger_channels=256,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(Hamburger, self).__init__()
        self.ham_in = nn.Sequential(
            nn.Conv2d(
                in_channels=hamburger_channels,
                out_channels=hamburger_channels,
                kernel_size=(1, 1)
            )
        )

        self.ham = bricks.NMF2D(args=nmf2d_config)

        self.ham_out = nn.Sequential(
            nn.Conv2d(
                in_channels=hamburger_channels,
                out_channels=hamburger_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=hamburger_channels
            )
        )

    def forward(self, x):
        out = self.ham_in(x)
        out = self.ham(out)
        out = self.ham_out(out)
        out = F.relu(x + out)
        return out


class LightHamHead(nn.Module):

    def __init__(
            self,
            num_classes,
            in_channels_list=[64, 160, 256],
            hidden_channels=256,
            out_channels=256,
            n_classes=150,
            drop_prob=0.1,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(LightHamHead, self).__init__()

        self.cls_seg = nn.Sequential(
            nn.Dropout2d(drop_prob),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=n_classes,
                kernel_size=(1, 1)
            )
        )

        self.squeeze = nn.Sequential(
            nn.Conv2d(
                in_channels=sum(in_channels_list),
                out_channels=hidden_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=hidden_channels,
            ),
            nn.ReLU()
        )

        self.hamburger = Hamburger(
            hamburger_channels=hidden_channels,
            nmf2d_config=nmf2d_config
        )

        self.align = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=out_channels
            ),
            nn.ReLU()
        )

    def forward(self, inputs):
        assert len(inputs) >= 2
        o = inputs[0]
        batch_size, _, standard_height, standard_width = inputs[1].shape
        standard_shape = (standard_height, standard_width)
        inputs = [
            F.interpolate(
                input=x,
                size=standard_shape,
                mode="bilinear",
                align_corners=False
            )
            for x in inputs[1:]
        ]
        x = torch.cat(inputs, dim=1)
        out = self.squeeze(x)
        out = self.hamburger(out)
        out = self.align(out)
        out = self.cls_seg(out)
        _, _, original_height, original_width = o.shape
        out = F.interpolate(
            input=out,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False
        )
        out = out.view(batch_size, -1, original_height, original_width)

        return out


class SegNeXt(nn.Module):

    def __init__(
            self,
            num_classes,
            embed_dims=[3, 32, 64, 160, 256],
            expand_ratios=[8, 8, 4, 4],
            depths=[3, 3, 5, 2],
            drop_prob_of_encoder=0.1,
            drop_path_prob=0.1,
            hidden_channels=256,
            out_channels=256,
            drop_prob_of_decoder=0.1,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(SegNeXt, self).__init__()

        # 共享编码器
        self.encoder = MSCAN(
            embed_dims=embed_dims,
            expand_ratios=expand_ratios,
            depths=depths,
            drop_prob=drop_prob_of_encoder,
            drop_path_prob=drop_path_prob
        )

        # 为每个任务创建独立的解码器和分类头部
        self.sic_decoder = LightHamHead(
            num_classes['SIC'],  # SIC 任务
            in_channels_list=embed_dims[-3:],
            hidden_channels=hidden_channels,
            out_channels=out_channels,

            drop_prob=drop_prob_of_decoder,
            nmf2d_config=nmf2d_config
        )

        self.sod_decoder = LightHamHead(
            num_classes['SOD'],  # SIC 任务
            in_channels_list=embed_dims[-3:],
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            drop_prob=drop_prob_of_decoder,
            nmf2d_config=nmf2d_config
        )

        self.floe_decoder = LightHamHead(
            num_classes['FLOE'],  # SIC 任务
            in_channels_list=embed_dims[-3:],
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            drop_prob=drop_prob_of_decoder,
            nmf2d_config=nmf2d_config
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out), mean=0)

    def forward(self, x):
        # 编码器提取特征
        enc_feats = self.encoder(x)

        # 每个任务独立的解码过程
        sic_output = self.sic_decoder(enc_feats)
        sod_output = self.sod_decoder(enc_feats)
        floe_output = self.floe_decoder(enc_feats)

        return sic_output, sod_output, floe_output
