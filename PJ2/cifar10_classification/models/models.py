import numpy as np
import torch
from torch import nn
from typing import Sequence, Tuple, Union
from utils.train_utils import get_activation


# ===================== 工具函数与初始化 =====================
def init_weights_(m, activation_name='relu'):
    # 激活函数到kaiming参数的映射
    nonlinearity_map = {
        'relu': 'relu',
        'leaky_relu': 'leaky_relu',
        'gelu': 'relu',
        'swish': 'sigmoid',
        'mish': 'tanh',
        'sigmoid': 'sigmoid',
        'tanh': 'tanh'
    }
    nonlinearity = nonlinearity_map.get(activation_name.lower(), 'relu')
    a = 0.1 if activation_name.lower() == 'leaky_relu' else 0.0

    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
            m.weight, mode='fan_out',
            nonlinearity=nonlinearity, a=a
        )
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.zeros_(m.bias)


def get_number_of_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ===================== 基础模型基类 =====================
class BaseModel(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, activation='relu', init_weights=True):
        super().__init__()
        self.activation_name = activation.lower()
        self.activation = get_activation(activation)
        self.inp_ch = inp_ch
        self.num_classes = num_classes
        self.init_weights_flag = init_weights

    def _make_classifier(self, in_features: int, fc_neurons: Union[int, Sequence[int]],
                         dropout_prob: float = 0.0) -> nn.Sequential:
        layers = []
        dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

        if isinstance(fc_neurons, int):
            fc_neurons = (fc_neurons, fc_neurons)

        for neurons in fc_neurons:
            layers.append(nn.Linear(in_features, neurons))
            layers.append(self.activation)
            layers.append(dropout)
            in_features = neurons

        layers.append(nn.Linear(in_features, self.num_classes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _apply_init_weights(self):
        """统一应用初始化"""
        if self.init_weights_flag:
            self.apply(lambda m: init_weights_(m, self.activation_name))


# ===================== 残差块 =====================
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 use_bn: bool = True, activation_name: str = 'relu'):
        super().__init__()
        self.use_bn = use_bn
        self.activation = get_activation(activation_name)

        # 主分支：所有接BN的卷积层去掉bias
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=not use_bn
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=3, stride=1, padding=1, bias=not use_bn
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion) if use_bn else nn.Identity()

        # 快捷分支：维度不匹配时用1x1卷积对齐
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=not use_bn
                ),
                nn.BatchNorm2d(out_channels * self.expansion) if use_bn else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.activation(out)
        return out


# ===================== 四个标准模型 =====================
class Original(BaseModel):
    """纯卷积基线模型（无BN、无残差）"""
    def __init__(self, inp_ch=3, num_classes=10, activation='relu',
                 conv_filters=64, fc_neurons=512, init_weights=True):
        super().__init__(inp_ch, num_classes, activation, init_weights)
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
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = self._make_classifier(cf*8, fc_neurons)
        self._apply_init_weights()


class Original_BN(BaseModel):
    """带BatchNorm的基线模型"""
    def __init__(self, inp_ch=3, num_classes=10, activation='relu',
                 conv_filters=64, fc_neurons=512, init_weights=True):
        super().__init__(inp_ch, num_classes, activation, init_weights)
        cf = conv_filters

        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, cf, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(cf), self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cf, cf*2, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(cf*2), self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cf*2, cf*4, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(cf*4), self.activation,
            nn.Conv2d(cf*4, cf*4, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(cf*4), self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cf*4, cf*8, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(cf*8), self.activation,
            nn.Conv2d(cf*8, cf*8, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(cf*8), self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cf*8, cf*8, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(cf*8), self.activation,
            nn.Conv2d(cf*8, cf*8, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(cf*8), self.activation,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = self._make_classifier(cf*8, fc_neurons)
        self._apply_init_weights()


class Original_BN_Dropout(Original_BN):
    """带BatchNorm和Dropout的基线模型"""
    def __init__(self, inp_ch=3, num_classes=10, activation='relu',
                 conv_filters=64, fc_neurons=512, dropout_prob=0.5, init_weights=True):
        super().__init__(inp_ch, num_classes, activation, conv_filters, fc_neurons, init_weights=False)
        cf = conv_filters

        self.classifier = self._make_classifier(cf*8, fc_neurons, dropout_prob)
        self._apply_init_weights()


class Original_Residual_BN(BaseModel):
    """带残差连接和BatchNorm的模型"""
    def __init__(self, inp_ch=3, num_classes=10, activation='relu',
                 conv_filters=64, fc_neurons=512, blocks_per_stage=(2, 2, 2, 2),
                 init_weights=True):
        super().__init__(inp_ch, num_classes, activation, init_weights)
        self.in_planes = conv_filters
        cf = conv_filters

        # 初始卷积层
        self.init_conv = nn.Sequential(
            nn.Conv2d(inp_ch, cf, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cf),
            self.activation
        )

        self.stage1 = self._make_layer(cf, blocks_per_stage[0], stride=1)
        self.stage2 = self._make_layer(cf*2, blocks_per_stage[1], stride=2)
        self.stage3 = self._make_layer(cf*4, blocks_per_stage[2], stride=2)
        self.stage4 = self._make_layer(cf*8, blocks_per_stage[3], stride=2)

        self.features = nn.Sequential(
            self.init_conv,
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = self._make_classifier(cf*8, fc_neurons)
        self._apply_init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """批量构建残差块"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(
                self.in_planes, planes, stride=s,
                use_bn=True, activation_name=self.activation_name
            ))
            self.in_planes = planes * ResidualBlock.expansion
        return nn.Sequential(*layers)


# ===================== 统一模型构建入口 =====================
def build_model(cfg: dict) -> nn.Module:
    model_name = cfg.get("model_name", "original")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    # 通用参数
    common_kwargs = {
        "num_classes": cfg.get("num_classes", 10),
        "activation": cfg.get("activation", "relu"),
        "conv_filters": cfg.get("conv_filters", 64),
        "fc_neurons": cfg.get("fc_neurons", 512),
        "init_weights": cfg.get("init_weights", True)
    }

    # 模型特定参数
    if model_name == "original_bn_dropout":
        common_kwargs["dropout_prob"] = cfg.get("dropout_prob", 0.5)
    elif model_name == "original_residual_bn":
        common_kwargs["blocks_per_stage"] = cfg.get("blocks_per_stage", (2, 2, 2, 2))

    return MODEL_REGISTRY[model_name](**common_kwargs)


# ===================== 模型注册 =====================
MODEL_REGISTRY = {
    'original': Original,
    'original_bn': Original_BN,
    'original_bn_dropout': Original_BN_Dropout,
    'original_residual_bn': Original_Residual_BN
}