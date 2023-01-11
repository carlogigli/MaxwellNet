# modified from # Adapted from https://github.com/jvanvugt/pytorch-unet, MIT License, Copyright (c) 2018 Joris

import torch
from torch import nn
import torch.nn.functional as F

class TUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, depth=5, wf=6, norm='weight', up_mode='upconv', groups=1, tune_channels=2**4):
        super(TUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.latent = nn.ModuleList()
        self.fc = nn.ModuleList()

        prev_channels = int(in_channels)
        self.ker_size = 3
        self.tune_channels = tune_channels
        self.tune_depth = 1
        self.ch_lat_in = wf * 2 ** (self.tune_depth) # V1.0: Tuner along encoder path
        self.ch_lat_out = wf * 2 ** (self.tune_depth) # V1.0: Tuner along encoder path

        for i in range(depth-1):

            if i == 0:
                self.encoder.append(
                    UNetConvBlock(prev_channels, [wf * (2 ** i), wf * (2 ** i)], self.ker_size, 0, norm, 1))
            elif i == self.tune_depth:
                self.encoder.append(UNetConv(prev_channels, wf * (2 ** i), self.ker_size, 0, norm, groups, nn.CELU()))
            else:
                self.encoder.append(
                    UNetConvBlock(prev_channels, [wf * (2 ** i), wf * (2 ** i)], self.ker_size, 0, norm, groups))
            prev_channels = int(wf * (2 ** i))
            self.encoder.append(nn.AvgPool2d(2))

        self.latent.append(UNetConv(prev_channels, wf * 2 ** (depth - 1), 3, 0, norm, groups, nn.CELU()))
        self.latent.append(UNetConv(wf * 2 ** (depth - 1), wf * 2 ** (depth - 2), 3, 0, norm, groups, None))

        prev_channels = int(wf * 2 ** (depth - 2))

        for i in reversed(range(depth - 1)):
            self.decoder.append(
                UNetUpBlock(prev_channels, [wf * (2 ** i), int(wf * (2 ** (i - 1)))], up_mode, 3, 0, norm, groups))
            prev_channels = int(wf * (2 ** (i - 1)))

        self.last_conv = nn.Conv2d(prev_channels, out_channels, kernel_size=1, padding=0, bias=True, groups=groups)

        for i in range(4):
            self.fc.append(FClayer(2**(2*i),2**(2*(i+1)),activation=nn.CELU()))
            prev_feat = int(2**(2*(i+1)))

        self.fc.append(FClayer(prev_feat, self.ch_lat_in*self.ch_lat_out*self.ker_size**2 + self.ch_lat_out, activation=None)) # V1.0: Tuner along encoder path

    def forward(self, input, tune_param):
        blocks = []
        x = input
        y = tune_param.unsqueeze(1).float()
        for fc in self.fc:
            y = fc(y)

        # V1.0: Tuner along encoder path
        K = torch.reshape(y[:, :-self.ch_lat_out],(x.shape[0],self.ch_lat_out,self.ch_lat_in,self.ker_size,self.ker_size))
        b = y[:, -self.ch_lat_out:]

        for i, down in enumerate(self.encoder):
            if i == 2 * self.tune_depth: # V1.0: Tuner along encoder path
                x = down(x)
                #x = torch.stack([corr2d_multi_in_out(x_, K_, b_) for x_, K_, b_ in zip(x, K, b)], 0)
                m = nn.ReplicationPad2d(1)
                x = torch.cat([F.conv2d(m(x_.unsqueeze(0)),K_, bias = b_) for x_, K_, b_ in zip(x, K, b)], dim = 0)
            else:
                x = down(x)
            if i % 2 == 0: # and i < (len(self.encoder) - 2):
                blocks.append(x)

        x = self.latent[0](x)
        latent = x
        x = self.latent[1](x)

        for i, up in enumerate(self.decoder):
            x = up(x, blocks[-i - 1])

        return self.last_conv(x), latent

class FClayer(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super(FClayer, self).__init__()
        block = []
        block.append(nn.Linear(in_features, out_features, bias=True))
        if activation is not None:
            block.append(activation)
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kersize, padding, norm, groups):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(UNetConv(in_size=in_size, out_size=out_size[0], kersize=kersize, padding=padding, norm=norm, groups=groups, activation=nn.CELU()))
        block.append(UNetConv(in_size=out_size[0], out_size=out_size[1], kersize=kersize, padding=padding, norm=norm, groups=groups, activation=None))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetConv(nn.Module):
    def __init__(self, in_size, out_size, kersize, padding, norm, groups, activation=None):
        super(UNetConv, self).__init__()
        block = []
        block.append(nn.ReplicationPad2d(1))
        if norm == 'weight':
            block.append(nn.utils.weight_norm((nn.Conv2d(in_size, out_size, kernel_size=int(kersize),
                                                         padding=int(0), bias=True, groups=groups)), name='weight'))
        elif norm == 'batch':
            block.append(nn.Conv2d(in_size, out_size, kernel_size=int(kersize),
                                   padding=int(padding), bias=True, groups=groups))
            block.append(nn.BatchNorm2d(out_size))

        elif norm == 'no':
            block.append((nn.Conv2d(in_size, out_size, kernel_size=int(kersize),
                                    padding=int(0), bias=True, groups=groups)))

        if activation is not None:
            block.append(activation)

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, kersize, padding, norm, groups):
        super(UNetUpBlock, self).__init__()
        block = []
        if up_mode == 'upconv':
            block.append(nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2, bias=False, groups=groups))
        elif up_mode == 'upsample':
            block.append(nn.Upsample(mode='bilinear', scale_factor=2))
            block.append(nn.Conv2d(in_size, in_size, kernel_size=1, bias=False, groups=groups))

        self.block = nn.Sequential(*block)
        self.conv_block = UNetConvBlock(in_size * 2, out_size, kersize, padding, norm, groups)
        self.groups = groups

    def forward(self, x, bridge):
        up = self.block(x)
        if self.groups == 1:
            out = torch.cat([up, bridge], 1)
        elif self.groups == 2:
            channels = up.shape[1]
            out = torch.cat((up[:, 0:channels // 2, :, :], bridge[:, 0:channels // 2, :, :],
                             up[:, channels // 2:channels, :, :], bridge[:, channels // 2:channels, :, :]), 1)
        elif self.groups == 3:
            channels = up.shape[1]
            out = torch.cat((up[:, 0:channels//3, :, :], bridge[:, 0:channels//3, :, :], up[:, channels//3:channels//3*2, :, :], bridge[:, channels//3:channels//3*2, :, :], up[:, channels//3*2:channels, :, :], bridge[:, channels//3*2:channels, :, :]), 1)
        out = self.conv_block(out)
        return out
