import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.train_utils import get_activation, init_weights_

def init_weights_(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_activation(name):
    if name == 'relu':
        return nn.ReLU(True)
    elif name == 'leaky_relu':
        return nn.LeakyReLU(0.1, True)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'swish':
        return nn.SiLU()
    elif name == 'mish':
        return nn.Mish()
    else:
        raise ValueError(f"Unsupported activation: {name}")

class Original(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, activation='relu', 
                 conv_filters=64, fc_neurons=512, init_weights=True):
        super().__init__()
        self.activation = get_activation(activation)
        cf = conv_filters
        
        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, cf, kernel_size=3, padding=1), self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cf, cf*2, kernel_size=3, padding=1), self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cf*2, cf*4, kernel_size=3, padding=1), self.activation,
            nn.Conv2d(cf*4, cf*4, kernel_size=3, padding=1), self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cf*4, cf*8, kernel_size=3, padding=1), self.activation,
            nn.Conv2d(cf*8, cf*8, kernel_size=3, padding=1), self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cf*8, cf*8, kernel_size=3, padding=1), self.activation,
            nn.Conv2d(cf*8, cf*8, kernel_size=3, padding=1), self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(cf*8, fc_neurons), self.activation,
            nn.Linear(fc_neurons, fc_neurons), self.activation,
            nn.Linear(fc_neurons, num_classes)
        )
        
        if init_weights:
            self.apply(init_weights_)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class Original_BN(Original):
    def __init__(self, inp_ch=3, num_classes=10, activation='relu',
                 conv_filters=64, fc_neurons=512, init_weights=True):
        super().__init__(inp_ch, num_classes, activation, conv_filters, fc_neurons, init_weights=False)
        cf = conv_filters
        act = self.activation
        
        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, cf, kernel_size=3, padding=1), nn.BatchNorm2d(cf), act,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cf, cf*2, kernel_size=3, padding=1), nn.BatchNorm2d(cf*2), act,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cf*2, cf*4, kernel_size=3, padding=1), nn.BatchNorm2d(cf*4), act,
            nn.Conv2d(cf*4, cf*4, kernel_size=3, padding=1), nn.BatchNorm2d(cf*4), act,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cf*4, cf*8, kernel_size=3, padding=1), nn.BatchNorm2d(cf*8), act,
            nn.Conv2d(cf*8, cf*8, kernel_size=3, padding=1), nn.BatchNorm2d(cf*8), act,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cf*8, cf*8, kernel_size=3, padding=1), nn.BatchNorm2d(cf*8), act,
            nn.Conv2d(cf*8, cf*8, kernel_size=3, padding=1), nn.BatchNorm2d(cf*8), act,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        if init_weights:
            self.apply(init_weights_)

class Original_BN_Dropout(Original_BN):
    def __init__(self, inp_ch=3, num_classes=10, activation='relu',
                 conv_filters=64, fc_neurons=512, dropout_prob=0.1, init_weights=True):
        super().__init__(inp_ch, num_classes, activation, conv_filters, fc_neurons, init_weights=False)
        cf = conv_filters
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(cf*8, fc_neurons), self.activation,
            nn.Dropout(dropout_prob),
            nn.Linear(fc_neurons, fc_neurons), self.activation,
            nn.Linear(fc_neurons, num_classes)
        )
        if init_weights:
            self.apply(init_weights_)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_bn=True, activation=nn.ReLU()):
        super().__init__()
        self.use_bn = use_bn
        self.activation = activation
        
        # 主分支
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # 快捷分支：当通道数或步长变化时，用1x1卷积调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out

class Original_Residual_BN(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, activation='relu',
                 conv_filters=64, fc_neurons=512, init_weights=True):
        super().__init__()
        self.activation = get_activation(activation)
        cf = conv_filters
        
        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, cf, kernel_size=3, padding=1), nn.BatchNorm2d(cf), self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cf, cf*2, kernel_size=3, padding=1), nn.BatchNorm2d(cf*2), self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(cf*2, cf*4, use_bn=True, activation=self.activation),
            ResidualBlock(cf*4, cf*4, use_bn=True, activation=self.activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(cf*4, cf*8, use_bn=True, activation=self.activation),
            ResidualBlock(cf*8, cf*8, use_bn=True, activation=self.activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(cf*8, cf*8, use_bn=True, activation=self.activation),
            ResidualBlock(cf*8, cf*8, use_bn=True, activation=self.activation),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(cf*8, fc_neurons), self.activation,
            nn.Linear(fc_neurons, fc_neurons), self.activation,
            nn.Linear(fc_neurons, num_classes)
        )
        
        if init_weights:
            self.apply(init_weights_)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters())

MODEL_REGISTRY = {
    'original': Original,
    'original_bn': Original_BN,
    'original_bn_dropout': Original_BN_Dropout,
    'original_residual_bn': Original_Residual_BN
}