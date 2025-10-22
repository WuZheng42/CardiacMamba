import sys
import os

# 将项目根目录添加到 Python 的搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.nn import LayerNorm

import torch.fft
from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
import math
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba

sys.path.append('/root/autodl-tmp/EquiPleth-main/nndl/mamba_main')

from nndl.losses.NegPearsonLoss import Neg_Pearson

from mamba_ssm import Mamba
from typing import Tuple
# Mamba
from mamba_ssm import Mamba
from nndl.mamba.bimamba import Mamba as BiMamba
from nndl.mamba.mm_bimamba import Mamba as MMBiMamba
from nndl.model.base import BaseNet




class MMMambaEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            d_ffn,
            activation='ReLU',
            dropout=0.0,
            causal=False,
            mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        if activation == 'ReLU':
            activation = torch.nn.ReLU
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = torch.nn.ReLU

        bidirectional = mamba_config.pop('bidirectional', True)  # 设置默认值为 True

        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = MMBiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

        self.a_downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(d_model),
        )

    def forward(
            self,
            a_x, v_x,
            a_inference_params=None,
            v_inference_params=None
    ):

        a_out1, v_out1 = self.mamba(a_x, v_x, a_inference_params, v_inference_params)
        # print(f"ax:{a_x.shape}")
        # print(f"a_out1:{a_out1.shape}")
        if a_x is not None and a_out1 is not None:
            a_out = a_x + self.norm1(a_out1)
        # a_out = self.norm1(a_out1)
        if v_x is not None and v_out1 is not None:
            v_out = v_x + self.norm2(v_out1)
        # v_out = self.norm2(v_out1)
        if a_x is not None and v_x is not None:
            return a_out, v_out
        if a_x is None:
            return a_x,v_out
        if v_x is None:
            return a_out,v_x


class MMCNNEncoderLayer(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            dropout=0.0,
            causal=False,
            dilation=1,
    ):
        super().__init__()

        self.a_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.a_bn = nn.BatchNorm1d(output_size)

        self.v_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.v_bn = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()

        self.a_drop = nn.Dropout(dropout)
        self.v_drop = nn.Dropout(dropout)

        self.a_net = nn.Sequential(self.a_conv, self.a_bn, self.relu, self.a_drop)
        self.v_net = nn.Sequential(self.v_conv, self.v_bn, self.relu, self.v_drop)

        if input_size != output_size:
            self.a_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
            self.v_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.a_skipconv = None
            self.v_skipconv = None

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.a_conv.weight.data)
        nn.init.xavier_uniform_(self.v_conv.weight.data)
        # nn.init.xavier_uniform_(self.conv2.weight.data)

    def forward(self, xa, xv):

        # print(f"xa{xa.shape}")
        # print(f"xv{xv.shape}")
        if xa is not None:
            a_out = self.a_net(xa)
        if xv is not None:
            v_out = self.v_net(xv)
        if xa is not None:
            if self.a_skipconv is not None:
                xa = self.a_skipconv(xa)
        if xv is not None:
            if self.v_skipconv is not None:
                xv = self.v_skipconv(xv)
        if xa is not None and a_out is not None:
            a_out = a_out + xa
        if xv is not None and v_out is not None:
            v_out = v_out + xv
        if xa is not None and xv is not None:
            return a_out, v_out
        if xv is None:
            return a_out,xv
        if xa is None:
            return xa, v_out

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x * attention


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super(MambaLayer, self).__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        drop_path = 0
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_patch_token(self, x):

        B, C, L = x.shape
        assert C == self.dim

        x_flat = x.transpose(-1, -2)  # Change to (B, L, C)
        x_norm = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm)
        x_out = self.norm2(x_flat + self.drop_path(x_mamba))
        out = x_out.transpose(-1, -2)  # Change back to (B, C, L)

        return out

    def forward(self, x):
        return self.forward_patch_token(x)


class Fusion_Stem(nn.Module):
    def __init__(self, apha=0.5, belta=0.5, dim=24):
        super(Fusion_Stem, self).__init__()

        self.stem11 = nn.Sequential(nn.Conv2d(3, dim // 2, kernel_size=7, stride=2, padding=3),
                                    nn.BatchNorm2d(dim // 2, eps=1e-05, momentum=0.1, affine=True,
                                                   track_running_stats=True),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True)
,
                                    nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
                                    )

        self.stem12 = nn.Sequential(nn.Conv2d(12, dim // 2, kernel_size=7, stride=2, padding=3),
                                    nn.BatchNorm2d(dim // 2),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True)
,
                                    nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
                                    )

        self.stem21 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
,
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.stem22 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
       # print("Initial input shape:", x.shape)
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,C,H/8,W/8]
        """
        N, C, D, H, W = x.shape
        x1 = torch.cat([x[:, :, :1, :, :], x[:, :, :1, :, :], x[:, :, :D - 2, :, :]], 2)
        x2 = torch.cat([x[:, :, :1, :, :], x[:, :, :D - 1, :, :]], 2)
        x3 = x
        x4 = torch.cat([x[:, :, 1:, :, :], x[:, :, D - 1:, :, :]], 2)
        x5 = torch.cat([x[:, :, 2:, :, :], x[:, :,D - 1 :, :, :], x[:, :, D - 1:, :, :]], 2)
        #print("Shape after concatenation:", torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 2).shape)

       # 查看拼接后的张量的形状

        # print("x2 - x1:", x2 - x1)
        # print("x3 - x2:", x3 - x2)
        # print("x4 - x3:", x4 - x3)
        # print("x5 - x4:", x5 - x4)

        x_diff = self.stem12(torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 3).reshape(N * D, 12, H, W))
       # 查看x_diff层的输出（激活值）

        # print("x_diff output shape:", x_diff.shape)
        # print("x_diff after GeLU mean:", x_diff.mean().item())
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)
       # 查看stem11输出的激活值

        # print("x after stem11 output shape:", x.shape)
        # print("x after stem11 mean:", x.mean().item())

        # fusion layer1
        x_path1 = self.apha * x + self.belta * x_diff
        x_path1 = self.stem21(x_path1)
       # 查看stem21的输出

        # print("x_path1 after stem21 output shape:", x_path1.shape)
        # print("x_path1 after stem21 mean:", x_path1.mean().item())
        # fusion layer2
        x_path2 = self.stem22(x_diff)

        # print("x_path2 after stem22 output shape:", x_path2.shape)
        # print("x_path2 after stem22 mean:", x_path2.mean().item())
        x = self.apha * x_path1 + self.belta * x_path2

        return x


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class RF_conv_encoder(torch.nn.Module):
    def __init__(self, channels=10):
        super(RF_conv_encoder, self).__init__()

        self.ConvBlock1 = torch.nn.Sequential(
            torch.nn.Conv1d(channels, 48, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(48),
            torch.nn.ReLU(),
        )

        self.ConvBlock2 = torch.nn.Sequential(
            torch.nn.Conv1d(48, 64, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 80, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(80),
            torch.nn.ReLU(inplace=True),
            MambaLayer(dim=80),
            ChannelAttention3D(in_channels=80, reduction=2),
        )
        self.ConvBlock4 = torch.nn.Sequential(
            torch.nn.Conv1d(80, 96, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(96),
            torch.nn.ReLU(inplace=True),
            MambaLayer(dim=96),
            ChannelAttention3D(in_channels=96, reduction=2),
        )
        self.ConvBlock5_mean = torch.nn.Sequential(
            torch.nn.Conv1d(64, 96, 7, stride=1, padding=3),
        )
        self.downsample1 = torch.nn.MaxPool1d(kernel_size=2)
        self.downsample2 = torch.nn.MaxPool1d(kernel_size=2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x) -> torch.tensor:
        N,C,D = x.shape
        x1 = torch.cat([x[:, :, :1], x[:, :, :1], x[:, :, :D - 2]], 2)
        x2 = torch.cat([x[:, :, :1], x[:, :, :D - 1]], 2)
        x3 = x
        x4 = torch.cat([x[:, :, 1:], x[:, :, D - 1:]], 2)
        x5 = torch.cat([x[:, :, 2:], x[:, :, D - 1:], x[:, :, D - 1:]], 2)
        x_diff = self.ConvBlock1(torch.cat([x2 - x1, x3 - x2, x4 - x3,x5 - x4], 1))


        x = self.ConvBlock2(x_diff)

        x = self.ConvBlock3(x)

        x = self.downsample1(x)
        x = self.ConvBlock4(x)
        x = self.downsample2(x)

        return x


class CoSSM(nn.Module):
    """This class implements the CoSSM encoder.
    """

    def __init__(
            self,
            num_layers,
            input_size,
            output_sizes=[256, 512, 512],
            d_ffn=1024,
            activation='Swish',
            dropout=0.0,
            kernel_size=3,
            causal=False,
            mamba_config=None
    ):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')
        prev_input_size = input_size

        cnn_list = []
        mamba_list = []
        # print(output_sizes)
        for i in range(len(output_sizes)):
            cnn_list.append(MMCNNEncoderLayer(
                input_size=input_size if i < 1 else output_sizes[i - 1],
                output_size=output_sizes[i],
                dropout=dropout
            ))
            mamba_list.append(MMMambaEncoderLayer(
                d_model=output_sizes[i],
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)

    def forward(
            self,
            a_x, v_x,
            a_inference_params=None,
            v_inference_params=None
    ):
        a_out = a_x
        v_out = v_x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            if a_out is not None:
                a_out = a_out.permute(0, 2, 1)
            if v_out is not None:
                v_out = v_out.permute(0, 2, 1)
            a_out, v_out = cnn_layer(a_out, v_out)
            if a_out is not None:
                a_out = a_out.permute(0, 2, 1)
            if v_out is not None:
                v_out = v_out.permute(0, 2, 1)
            a_out, v_out = mamba_layer(
                a_out, v_out,
                a_inference_params=a_inference_params,
                v_inference_params=v_inference_params
            )

        return a_out, v_out

class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()

        self.scale = 0.02
        self.dim = dim * mlp_ratio

        self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim))

        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm1d(dim * mlp_ratio),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(dim * mlp_ratio, dim, 1, 1, 0, bias=False),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        B, N, C = x.shape

        x = self.fc1(x).transpose(1, 2)

        x_fre = torch.fft.fft(x, dim=1, norm='ortho')  # FFT on N dimension

        x_real = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.real, self.r) - \
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.i) + \
            self.rb
        )
        x_imag = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.r) + \
            torch.einsum('bnc,cc->bnc', x_fre.real, self.i) + \
            self.ib
        )

        x_fre = torch.stack([x_real, x_imag], dim=-1).float()
        x_fre = torch.view_as_complex(x_fre)
        x = torch.fft.ifft(x_fre, dim=1, norm="ortho")
        x = x.to(torch.float32)


        x = self.fc2(x.transpose(1, 2)).transpose(1, 2)
        return x





class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)      # 第一个卷积
        out = self.bn1(out)      # 第一个批归一化
        out = self.relu(out)     # ReLU 激活

        out = self.conv2(out)    # 第二个卷积
        out = self.bn2(out)      # 第二个批归一化

        if self.downsample is not None:
            identity = self.downsample(x)  # 如果尺寸不匹配，调整输入

        out += identity          # 残差连接
        out = self.relu(out)     # ReLU 激活

        return out

class RGB_extra(nn.Module):
    def __init__(self, input_channels=512):
        super(RGB_extra, self).__init__()
        self.layer1 = self._make_layer(input_channels, 256, blocks=2)
        self.layer2 = self._make_layer(256, 128, blocks=2)
        self.layer3 = self._make_layer(128, 64, blocks=2)
        self.layer4 = self._make_layer(64, 96, blocks=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock1D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入形状：x [B, 512, T]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class RF_extra(nn.Module):
    def __init__(self, input_channels=512):
        super(RF_extra, self).__init__()
        self.layer1 = self._make_layer(input_channels, 256, blocks=2)
        self.layer2 = self._make_layer(256, 128, blocks=2)
        self.layer3 = self._make_layer(128, 64, blocks=2)
        self.layer4 = self._make_layer(64, 96, blocks=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock1D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入形状：x [B, 512, T]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class RF_conv_decoder(torch.nn.Module):
    def __init__(self, channels=10):
        super(RF_conv_decoder, self).__init__()

        self.IQ_encoder = RF_conv_encoder(channels)

        self.ConvBlock1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(96, 48, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(48, 24, 7,stride=1, padding=3),
            torch.nn.BatchNorm1d(24),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x_IQ: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        z_IQ = self.IQ_encoder(x_IQ)
        x = self.ConvBlock1(z_IQ)
        x = self.ConvBlock2(x)

class CNN3D_extra(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvBlock5 = nn.Sequential(
            torch.nn.ConvTranspose1d(96, 48, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            torch.nn.ConvTranspose1d(48, 24, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(24),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.ConvBlock5(x)
        x = self.ConvBlock6(x)
        return x





class Decoder(nn.Module):
    def __init__(self, input_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv_final = nn.Conv1d(64, 1, kernel_size=1)
        self.upsample = nn.ConvTranspose1d(1, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # 输入形状 x: [B, C, T, 1, 1]
        # x = x.squeeze(-1).squeeze(-1)  # 形状变为 [B, C, T]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv_final(x)  # 形状变为 [B, 1, T]
        # x = self.upsample(x)  # 形状变为 [B, 1, 2*T]

        rPPG = x.squeeze(1)  # 输出形状为 [B, T]
        return rPPG


class FusionNet(nn.Module):
    def __init__(self, mamba_config=None):
        super().__init__()

        self.embed_dim = 96  # 定义 embed_dim
        self.Fusion_Stem = Fusion_Stem()
        self.stem3 = nn.Sequential(
            nn.Conv3d(24, self.embed_dim, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2)),
            nn.BatchNorm3d(self.embed_dim),
        )
        self.attn_mask = Attention_mask()
        self.RF_Stem = RF_conv_encoder(channels=40)
        self.CoSSM = CoSSM(
            num_layers=2,
            input_size=self.embed_dim,
            output_sizes=[256, 512, 512],
            d_ffn=1024,
            activation='ReLU',
            dropout=0.0,
            kernel_size=3,
            causal=False,
            mamba_config=mamba_config  # 确保传递正确的 mamba_config
        )
        self.FFT_block1 = Frequencydomain_FFN(dim=64, mlp_ratio=4)
        self.FFT_block2 = Frequencydomain_FFN(dim=64, mlp_ratio=4)
        self.RGB_extra = RGB_extra(input_channels=96)
        self.RF_extra = RF_extra(input_channels=96)
        self.decoder = Decoder(input_channels=512)
        self.reduce_channels = nn.Sequential(
            nn.Conv1d(512, 96, kernel_size=1),
            nn.BatchNorm1d(96),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=96, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=96, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU()
        )

    def forward(self, rgb_features, rf_features):
        if rgb_features is not None:
            N, C, D, H, W = rgb_features.shape
        # print("rgb input:", rgb.shape)
        # if rf_features is not None:
        #     N,C,D = rf_features.shape    #[32,10,128]
        if rgb_features is not None:
            rgb_features = self.Fusion_Stem(rgb_features)  # [N*D C H/8 W/8]
        if rgb_features is not None:
            rgb_features = rgb_features.view(N, D, self.embed_dim // 4, H // 8, W // 8).permute(0, 2, 1, 3, 4)

            rgb_features = self.stem3(rgb_features)    #[N,embedim,D,H/8,W/8]
        # print("before_rgb1:", rgb_features.shape)
            mask = torch.sigmoid(rgb_features)
            mask = self.attn_mask(mask)
            rgb_features = rgb_features * mask    #[N,embedim,D,H/8,W/8]
        # print("before_rgb2:", rgb_features.shape)
            rgb_features = torch.mean(rgb_features, 4)
            rgb_features = torch.mean(rgb_features, 3)    #[N,embedim,T]
        # print("before_rgb:", rgb_features.shape)
        # print("before_rgb3:", rgb_features.shape)
            rgb_features = rearrange(rgb_features, 'b c t -> b t c')
        if rf_features is not None:
            rf_features = self.RF_Stem(rf_features)              #[N,embedim,T]

        # print("rf_shape", rf_features.shape)
            rf_features = rearrange(rf_features, 'b c t -> b t c')

        # print("before_rgb:",rgb_features.shape)

        #rgb_features = self.reduce_channels(rgb_features)
        # print("after_rgb:",rgb_features.shape)
        # print("rf_shape",rf_features.shape)
        rgb_features, rf_features = self.CoSSM(rgb_features, rf_features)    #(32,128,512)即[B,T,C]

        # print("rgb shape cossm", rgb_features.shape)
        # print("rf shape cossm", rf_features.shape)
        # rgb_features = rearrange(rgb_features, 'b t c -> b c t')
        # rf_features = rearrange(rf_features, 'b t c -> b c t')
        #目前到上一行为止有梯度
        # rgb_features = self.reduce_channels(rgb_features)
        # rf_features = self.reduce_channels(rf_features)
        #print("rgb shape reduce", rgb_features.shape)
        #print("rf shape reduce", rf_features.shape)

        # rgb_features = self.conv1(rgb_features)
        # rf_features = self.conv2(rf_features)
        # rgb_features = rearrange(rgb_features, 'b c t -> b t c')
        # rf_features = rearrange(rf_features, 'b c t -> b t c')
        if rgb_features is not None:
            rgb_features = self.FFT_block1(rgb_features)
        if rf_features is not None:
            rf_features = self.FFT_block2(rf_features)     #(B,96,T)
        # print("rgb shape fft", rgb_features.shape)
        # print("rf shape fft", rf_features.shape)
        # 目前到上一行为止有梯度
        # rgb_features, rf_features = self.CoSSM(rgb_features, rf_features)     #(B,512,T)

        # rgb_features = self.RGB_extra(rgb_features)
        # rf_features = self.RF_extra(rf_features)         #(B,96,T)
        # 目前到上一行为止有梯度
        # rgb_features, rf_features = self.CoSSM(rgb_features, rf_features)    #(B,512,T)
        #print("rgb shape cossm2", rgb_features.shape)
        #print("rf shape cossm2", rf_features.shape)
        # 目前到上一行为止有梯度
        if rgb_features is not None and rf_features is not None:
            fused_features = rgb_features + rf_features       #(B,512,T)
        if rf_features is None:
            fused_features = rgb_features
        if rgb_features is None:
            fused_features = rf_features


        # fused_features = rearrange(fused_features, 'b t c -> b c t')
        # print("fused_features shape ", fused_features.shape)
        output = self.decoder(fused_features)
        # print("output shape ", output.shape)
        return output












