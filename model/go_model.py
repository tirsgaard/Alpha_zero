import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from functools import partial

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)      
def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels
    
class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels
    
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))

class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )
        
class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=17, blocks_sizes=[256], deepths=[0], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation))
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x
    
class policyHead(nn.Module):
    def __init__(self, filter_size=256, board_size=9, 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.board_size  = board_size
        
        self.conv_block = nn.Sequential(
                nn.Conv2d(filter_size, 2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(2),
                activation_func(activation))
        self.full_layer = nn.Sequential(nn.Linear(board_size*board_size*2, board_size*board_size+1, bias=True),
                                       activation_func(activation), nn.Softmax(dim=1))
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(-1,self.board_size*self.board_size*2)
        x = self.full_layer(x)
        return x


class valueHead(nn.Module):
    def __init__(self, filter_size=256, board_size=9, 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.board_size = board_size
        self.conv_block = nn.Sequential(
                nn.Conv2d(filter_size, 1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1),
                activation_func(activation))
        self.ff_layer1 = nn.Sequential(nn.Linear(board_size*board_size, filter_size, bias=True),
                                       activation_func(activation), nn.BatchNorm1d(filter_size))
        self.ff_layer2 = nn.Linear(filter_size,1)
        self.tanh_layer = nn.Tanh()
    
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(-1,self.board_size*self.board_size)
        x = self.ff_layer1(x)
        x = self.ff_layer2(x)
        x = self.tanh_layer(x)
        return x
    
    
class ResNet(nn.Module):
    
    def __init__(self, in_channels, filter_size, board_size, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, blocks_sizes=[filter_size], *args, **kwargs)
        self.policyHead = policyHead(filter_size, board_size, *args, **kwargs)
        self.valueHead = valueHead(filter_size, board_size, *args, **kwargs)
        
    def forward(self, x):
        x = self.encoder(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return [policy,value]