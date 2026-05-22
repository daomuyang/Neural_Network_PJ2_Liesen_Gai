import numpy as np
import torch
import os
from torch import nn
import random 


def get_accuracy(model, dataloader, device):
    """计算模型在数据集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    accuracy = 100 * correct / total if total > 0 else 0.0  # 防止除零
    model.train()
    return accuracy


def get_loss(model, dataloader, criterion, device):
    """计算模型在数据集上的平均损失"""
    model.eval()
    total_loss = 0.0
    total_batches = len(dataloader)
    if total_batches == 0:
        return 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    avg_loss = total_loss / total_batches
    model.train()
    return avg_loss


def set_random_seeds(seed_value=0, device='cpu'):
    """固定随机种子"""
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    if device_obj.type == 'cuda':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device_obj.type == 'mps':
        torch.mps.manual_seed(seed_value)


def create_experiment_dirs(experiment_name):
    """创建任务二专属实验目录"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_root = os.path.join(project_root, 'reports', experiment_name)
    exp_figures = os.path.join(exp_root, 'figures')
    exp_models = os.path.join(exp_root, 'models')
    exp_metrics = os.path.join(exp_root, 'metrics')
    
    # 递归创建目录，确保层级存在
    os.makedirs(exp_figures, exist_ok=True, mode=0o755)
    os.makedirs(exp_models, exist_ok=True, mode=0o755)
    os.makedirs(exp_metrics, exist_ok=True, mode=0o755)
    
    return exp_figures, exp_models, exp_metrics


def get_middle_conv_layer(model):
    """获取倒数第二个卷积层"""
    conv_layers = []
    if not hasattr(model, 'features'):
        raise ValueError("模型必须包含 'features' 模块（VGG结构）")
    
    for name, module in model.features.named_children():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(module)
    
    if len(conv_layers) < 2:
        raise ValueError(f"模型卷积层数量不足（仅找到 {len(conv_layers)} 个）")
    
    return conv_layers[-2]