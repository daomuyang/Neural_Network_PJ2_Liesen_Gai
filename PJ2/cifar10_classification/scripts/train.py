import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import argparse
import yaml
from tqdm import tqdm
import multiprocessing

from utils.custom_optimizers import CustomSGD, CustomAdam
from utils.train_utils import (
    FocalLoss, compute_regularization_loss,
    get_accuracy, get_loss, set_random_seeds, 
    save_experiment_results
)
from models.models import MODEL_REGISTRY
from data.loaders import get_cifar_loader

if sys.platform != 'win32':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass
else:
    multiprocessing.set_start_method('spawn', force=True)


def train(model, optimizer, criterion, train_loader, val_loader, device,
          exp_figures, exp_models, epochs_n=100, args=None,scheduler=None):
    model.to(device)
    train_loss_curve = np.zeros(epochs_n)
    val_loss_curve = np.zeros(epochs_n)
    train_acc_curve = np.zeros(epochs_n)
    val_acc_curve = np.zeros(epochs_n)
    max_val_acc = 0.0
    best_model_path = os.path.join(exp_models, 'best_model.pth')
    losses_list = []
    batches_n = len(train_loader)
    desc = f'Training {args.model}' if args else 'Training'
    
    for epoch in tqdm(range(epochs_n), unit='epoch', desc=desc):
        if scheduler is not None:
            scheduler.step()
        model.train()
        epoch_train_loss = 0.0
        step_losses = []
        
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)

            reg_loss = compute_regularization_loss(model, args.reg_type, args.weight_decay)
            total_loss = loss + reg_loss
            
            step_losses.append(total_loss.item())
            total_loss.backward()
            optimizer.step()
            epoch_train_loss += total_loss.item()
        
        # 统计指标
        train_loss_curve[epoch] = epoch_train_loss / batches_n
        val_loss_curve[epoch] = get_loss(model, val_loader, criterion, device)
        train_acc_curve[epoch] = get_accuracy(model, train_loader, device)
        val_acc_curve[epoch] = get_accuracy(model, val_loader, device)
        
        # 保存最优模型
        if val_acc_curve[epoch] > max_val_acc:
            max_val_acc = val_acc_curve[epoch]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': max_val_acc,
                'val_loss': val_loss_curve[epoch],
            }, best_model_path)
            tqdm.write(f"New best model saved! Val Acc: {max_val_acc:.2f}%")
        
        losses_list.append(step_losses)
        
        # 实时绘制训练曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1,2,1)
        plt.plot(train_loss_curve[:epoch+1], 'b-', label='Train Loss', linewidth=2)
        plt.plot(val_loss_curve[:epoch+1], 'orange', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1,2,2)
        plt.plot(train_acc_curve[:epoch+1], 'r-', label='Train Acc', linewidth=2)
        plt.plot(val_acc_curve[:epoch+1], 'g-', label='Val Acc', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_figures, 'training_curve.png'), dpi=300)
        plt.close()
    
    return losses_list, train_loss_curve, val_loss_curve, train_acc_curve, val_acc_curve, max_val_acc

def save_experiment_config(exp_dir, args, exp_id):
    config_path = os.path.join(exp_dir, "config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(f"Experiment ID: {exp_id}\n")
        f.write(f"Experiment Name: {args.name}\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Regularization Type: {args.reg_type}\n")
        f.write(f"Label Smoothing: {args.label_smoothing}\n")
        f.write(f"Conv Filters: {args.conv_filters}\n")
        f.write(f"FC Neurons: {args.fc_neurons}\n")
        f.write(f"Dropout Prob: {args.dropout_prob}\n")
        f.write(f"Focal Gamma: {args.focal_gamma}\n")
        f.write(f"Loss Type: {args.loss_type}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Num Workers: {args.num_workers}\n")
        f.write(f"Note: {args.note}\n")
    print(f"✅ Config saved to: {config_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 Classification Experiment')
    parser.add_argument('--exp', type=str, required=True, help='Experiment ID from configs/experiments.yaml')
    parser.add_argument('--config', type=str, default='configs/experiments.yaml', help='Config file path')
    parser.add_argument('--model', type=str, choices=MODEL_REGISTRY.keys())
    parser.add_argument('--experiment', type=str, help='Override experiment name')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--activation', type=str, choices=['relu', 'leaky_relu', 'gelu', 'sigmoid', 'tanh', 'swish', 'mish'])
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw', 'rmsprop', 'adagrad', 'adadelta', 'custom_sgd', 'custom_adam'])
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--label_smoothing', type=float)
    parser.add_argument('--conv_filters', type=int)
    parser.add_argument('--fc_neurons', type=int)
    parser.add_argument('--dropout_prob', type=float)
    parser.add_argument('--reg_type', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--loss_type', type=str, default='cross_entropy', choices=['cross_entropy', 'focal'])
    parser.add_argument('--focal_gamma', type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 实验ID转换
    try:
        exp_id = int(args.exp)
    except ValueError:
        raise ValueError(f"实验ID必须是数字，你输入的是: {args.exp}")
    
    if exp_id not in config['experiments']:
        available_ids = sorted(config['experiments'].keys())
        raise ValueError(f"实验ID {args.exp} 不存在！可用的ID: {available_ids}")
    
    exp_config = config['experiments'][exp_id]
    merged_args = config['defaults'].copy()
    merged_args.update(exp_config)
    
    import sys
    explicit_args = set()
    for i in range(1, len(sys.argv)):
        if sys.argv[i].startswith('--'):
            arg_name = sys.argv[i].lstrip('-').replace('-', '_')
            explicit_args.add(arg_name)
    for k, v in vars(args).items():
        if k in explicit_args and k not in ['exp', 'config']:
            if k == 'experiment':
                merged_args['name'] = v
            else:
                merged_args[k] = v
    
    args = argparse.Namespace(**merged_args)
    
    # 设备检测
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device('cpu')
        device_name = 'CPU'
    
    # 打印实验信息
    print("="*60)
    print(f"Experiment ID: {exp_id}")
    print(f"Experiment Name: {args.name}")
    print(f"Model: {args.model}")
    print(f"Device: {device} ({device_name})")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch Size: {args.batch_size}")
    print(f"Activation: {args.activation}, Optimizer: {args.optimizer}")
    print(f"Weight Decay: {args.weight_decay}, Label Smoothing: {args.label_smoothing}")
    print(f"Note: {args.note}")
    print("="*60)
    
    # 创建目录结构
    exp_dir = os.path.join('reports', args.name)
    exp_figures = os.path.join(exp_dir, 'figures')
    exp_metrics = os.path.join(exp_dir, 'metrics')
    exp_models = os.path.join(exp_dir, 'models')
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(exp_figures, exist_ok=True)
    os.makedirs(exp_metrics, exist_ok=True)
    os.makedirs(exp_models, exist_ok=True)
    
    # 随机种子设置
    set_random_seeds(int(args.seed), device)
    
    # 数据加载
    train_loader = get_cifar_loader(
        train=True, batch_size=int(args.batch_size), num_workers=int(args.num_workers))
    val_loader = get_cifar_loader(
        train=False, batch_size=int(args.batch_size), num_workers=int(args.num_workers))
    
    # 模型初始化
    model_class = MODEL_REGISTRY[args.model]
    model_kwargs = {
        'activation': args.activation,
        'conv_filters': int(args.conv_filters),
        'fc_neurons': int(args.fc_neurons),
    }
    if args.model == 'original_bn_dropout':
        model_kwargs['dropout_prob'] = float(args.dropout_prob)
    
    model = model_class(**model_kwargs)
    
    # 优化器
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=float(args.lr), 
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay)
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=float(args.lr), 
            weight_decay=float(args.weight_decay)
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=float(args.lr), 
            weight_decay=float(args.weight_decay)
        )
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=float(args.lr), 
            weight_decay=float(args.weight_decay)
        )
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(), 
            lr=float(args.lr), 
            weight_decay=float(args.weight_decay)
        )
    elif args.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), 
            lr=float(args.lr), 
            weight_decay=float(args.weight_decay)
        )
    elif args.optimizer == 'custom_sgd':
        optimizer = CustomSGD(
            model.parameters(), 
            lr=float(args.lr), 
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay)
        )
    elif args.optimizer == 'custom_adam':
        optimizer = CustomAdam(
            model.parameters(), 
            lr=float(args.lr), 
            betas=(0.9, 0.999), 
            weight_decay=float(args.weight_decay)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
        
    if isinstance(optimizer, (torch.optim.Optimizer,)):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    else:
        scheduler = None

    if args.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
        print(f"✅ 损失函数加载成功：CrossEntropyLoss | 标签平滑 = {args.label_smoothing}")
    elif args.loss_type == 'focal':
        criterion = FocalLoss(
            gamma=float(args.focal_gamma),
            label_smoothing=float(args.label_smoothing),
            num_classes=10
        )
        print(f"✅ 损失函数加载成功：FocalLoss | gamma = {args.focal_gamma} | 标签平滑 = {args.label_smoothing}")
    else:
        raise ValueError(f"不支持的损失函数类型: {args.loss_type}")
    
    # 开始训练
    losses_list, train_loss, val_loss, train_acc, val_acc, best_acc = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        exp_figures=exp_figures,
        exp_models=exp_models,
        epochs_n=int(args.epochs),
        args=args,
        scheduler=scheduler
    )
    
    # 保存结果
    save_experiment_config(exp_dir, args, exp_id)
    save_experiment_results(exp_metrics, args, losses_list, train_loss, val_loss, train_acc, val_acc, best_acc, model)
    
    print("\n" + "="*60)
    print("Experiment Completed!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {exp_dir}")
    print("="*60)