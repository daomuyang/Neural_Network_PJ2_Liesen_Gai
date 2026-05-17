"""
通用训练工具函数
"""
import numpy as np
import torch
import os
from torch import nn
import torch.nn.functional as F

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
        return nn.ReLU(inplace=True)
    elif name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
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
def get_accuracy(model, dataloader, device):
    """计算模型在数据集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    accuracy = 100 * correct / total
    model.train()
    return accuracy

def get_loss(model, dataloader, criterion, device):
    """计算模型在数据集上的平均损失"""
    model.eval()
    total_loss = 0.0
    total_batches = len(dataloader)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    avg_loss = total_loss / total_batches
    model.train()
    return avg_loss

def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if device.type == 'cuda': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_experiment_dirs(experiment_name):
    """创建实验目录结构"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_root = os.path.join(project_root, 'reports', experiment_name)
    exp_figures = os.path.join(exp_root, 'figures')
    exp_models = os.path.join(exp_root, 'models')
    exp_metrics = os.path.join(exp_root, 'metrics')
    os.makedirs(exp_figures, exist_ok=True)
    os.makedirs(exp_models, exist_ok=True)
    os.makedirs(exp_metrics, exist_ok=True)
    return exp_figures, exp_models, exp_metrics

def get_number_of_parameters(model):
    """计算模型的总参数量"""
    return sum(np.prod(p.shape).item() for p in model.parameters())

def save_experiment_results(exp_metrics, args, losses_list, train_loss, val_loss, train_acc, val_acc, best_acc, model):
    """统一保存所有实验结果和配置"""
    # 保存step级损失
    flat_losses = [loss for epoch_loss in losses_list for loss in epoch_loss]
    np.savetxt(os.path.join(exp_metrics, 'losses.txt'), flat_losses, fmt='%.6f')
    
    # 保存epoch级训练曲线
    np.savez(os.path.join(exp_metrics, 'training_curves.npz'),
             train_loss=train_loss, val_loss=val_loss, 
             train_acc=train_acc, val_acc=val_acc)
    
    # 保存实验配置和最终结果
    with open(os.path.join(exp_metrics, 'config.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nBest Validation Accuracy: {best_acc:.2f}%\n")
        f.write(f"Number of Parameters: {get_number_of_parameters(model):,}\n")

class FocalLoss(nn.Module):
    """手写Focal Loss"""
    def __init__(self, gamma=2.0, label_smoothing=0.0, num_classes=10):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        
        # 计算log_softmax（避免数值下溢）
        log_probs = F.log_softmax(inputs, dim=1)
        # 计算softmax概率
        probs = torch.exp(log_probs)
        
        # 计算逐样本的交叉熵损失
        one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).float()
        sample_ce_loss = -torch.sum(one_hot_targets * log_probs, dim=1)
        
        # 计算Focal权重:(1 - pt)^gamma
        pt = torch.sum(one_hot_targets * probs, dim=1)
        focal_weight = torch.pow(1 - pt, self.gamma)
        
        # Focal Loss = 平均(逐样本CE * Focal权重)
        focal_loss = torch.mean(focal_weight * sample_ce_loss)
        
        return focal_loss

def compute_regularization_loss(model, reg_type='l2', weight_decay=0.0):
    weight_decay = float(weight_decay)
    
    if weight_decay == 0 or reg_type not in ['l1', 'l2']:
        return 0.0
    
    reg_loss = 0.0
    for p in model.parameters():
        if reg_type == 'l1':
            reg_loss += torch.sum(torch.abs(p))
        else:  # l2
            reg_loss += torch.sum(torch.square(p))
    
    return weight_decay * reg_loss