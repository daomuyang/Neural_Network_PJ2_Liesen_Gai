import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import argparse
import yaml
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='numpy')


from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader
from utils.train_utils import (
    get_accuracy, get_loss, set_random_seeds,
    create_experiment_dirs, get_middle_conv_layer
)


MODEL_REGISTRY = {
    'VGG_A': VGG_A,
    'VGG_A_BatchNorm': VGG_A_BatchNorm
}


def train(model, optimizer, criterion, train_loader, val_loader, device,
          exp_figures, exp_models, epochs_n=100, scheduler=None):
    model.to(device)
    train_loss_curve = np.zeros(epochs_n)
    val_loss_curve = np.zeros(epochs_n)
    train_acc_curve = np.zeros(epochs_n)
    val_acc_curve = np.zeros(epochs_n)
    max_val_acc = 0.0
    best_model_path = os.path.join(exp_models, 'best_model.pth')
    losses_list = []
    grad_last_layer_avg = np.zeros(epochs_n)
    grad_mid_layer_avg = np.zeros(epochs_n)
    batches_n = len(train_loader)
    mid_conv_layer = get_middle_conv_layer(model)
    print(f"找到中间卷积层: {mid_conv_layer}")

    for epoch in tqdm(range(epochs_n), unit='epoch', desc=f'Training {model.__class__.__name__}'):
        if scheduler is not None:
            scheduler.step()
        
        model.train()
        epoch_train_loss = 0.0
        step_losses = []
        epoch_grad_last_sum = 0.0
        epoch_grad_mid_sum = 0.0

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            
            step_losses.append(loss.item())
            loss.backward()

            # 统计最后一层梯度范数
            grad_last_norm = torch.norm(model.classifier[-1].weight.grad).item()
            epoch_grad_last_sum += grad_last_norm
            
            # 统计中间层梯度范数
            grad_mid_norm = torch.norm(mid_conv_layer.weight.grad).item()
            epoch_grad_mid_sum += grad_mid_norm
            
            optimizer.step()
            epoch_train_loss += loss.item()

        # 计算epoch级指标
        train_loss_curve[epoch] = epoch_train_loss / batches_n
        val_loss_curve[epoch] = get_loss(model, val_loader, criterion, device)
        train_acc_curve[epoch] = get_accuracy(model, train_loader, device)
        val_acc_curve[epoch] = get_accuracy(model, val_loader, device)
        
        grad_last_layer_avg[epoch] = epoch_grad_last_sum / batches_n
        grad_mid_layer_avg[epoch] = epoch_grad_mid_sum / batches_n

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
            tqdm.write(f"New best model saved! Val Acc: {max_val_acc:.2f}%, Val Loss: {val_loss_curve[epoch]:.4f}")
        
        losses_list.append(step_losses)

        # 实时绘制训练曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1,2,1)
        plt.plot(train_loss_curve[:epoch+1], 'b-', label='Train Loss', linewidth=2, alpha=0.8)
        plt.plot(val_loss_curve[:epoch+1], 'orange', label='Val Loss', linewidth=2, alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1,2,2)
        plt.plot(train_acc_curve[:epoch+1], 'r-', label='Train Acc', linewidth=2, alpha=0.8)
        plt.plot(val_acc_curve[:epoch+1], 'g-', label='Val Acc', linewidth=2, alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_figures, 'training_curve.png'), dpi=300)
        plt.close()

    return losses_list, grad_last_layer_avg, grad_mid_layer_avg, train_loss_curve, val_loss_curve, train_acc_curve, val_acc_curve, max_val_acc

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    from multiprocessing import freeze_support
    freeze_support()

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='VGG BatchNorm Loss Landscape Experiment')
    parser.add_argument('--exp', type=str, required=True, help='Experiment ID from configs/experiments.yaml')
    parser.add_argument('--config', type=str, default='configs/experiments.yaml', help='Config file path')
    parser.add_argument('--model', type=str, choices=MODEL_REGISTRY.keys())
    parser.add_argument('--experiment', type=str, help='Override experiment name')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_workers', type=int)
    
    args = parser.parse_args()

    # 读取YAML配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 解析实验ID
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
    
    for k, v in vars(args).items():
        if v is not None and k not in ['exp', 'config']:
            if k == 'experiment':
                merged_args['name'] = v
            else:
                merged_args[k] = v
    
    args = argparse.Namespace(**merged_args)

    # 设备检测
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'Apple Silicon MPS'
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
    print(f"Note: {args.note}")
    print("="*60)

    # 实验初始化
    exp_figures, exp_models, exp_metrics = create_experiment_dirs(args.name)
    set_random_seeds(int(args.seed), device)

    # 数据加载
    train_loader = get_cifar_loader(
        train=True, batch_size=int(args.batch_size), num_workers=int(args.num_workers))
    val_loader = get_cifar_loader(
        train=False, batch_size=int(args.batch_size), num_workers=int(args.num_workers))

    # 模型初始化
    model_class = MODEL_REGISTRY[args.model]
    model = model_class()

    # 优化器与损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))

    # 余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs))

    # 开始训练
    losses_list, grad_last_layer_avg, grad_mid_layer_avg, train_loss, val_loss, train_acc, val_acc, best_acc = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        exp_figures=exp_figures,
        exp_models=exp_models,
        epochs_n=int(args.epochs),
        scheduler=scheduler
    )

    # 保存所有实验结果
    print("\n正在保存数据...")
    flat_losses = [loss for epoch_loss in losses_list for loss in epoch_loss]
    np.savetxt(os.path.join(exp_metrics, 'losses.txt'), flat_losses, fmt='%.6f')
    np.save(os.path.join(exp_metrics, 'grad_last_layer_avg.npy'), grad_last_layer_avg)
    np.save(os.path.join(exp_metrics, 'grad_mid_layer_avg.npy'), grad_mid_layer_avg)
    
    np.savez(os.path.join(exp_metrics, 'training_curves.npz'),
             train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc)

    # 自动保存实验配置到 config.txt
    def save_experiment_config(exp_root, args, exp_id):
        config_path = os.path.join(exp_root, "config.txt")
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
            f.write(f"Label Smoothing: {args.label_smoothing}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Num Workers: {args.num_workers}\n")
            f.write(f"Loss Type: {args.loss_type}\n")
            f.write(f"Note: {args.note}\n")
        print(f"✅ Experiment config saved to: {config_path}")

    # 调用保存函数
    exp_root = os.path.dirname(exp_figures)
    save_experiment_config(exp_root, args, exp_id)

    # 实验总结
    print("\n" + "="*60)
    print("Experiment Completed!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Final Train Accuracy: {train_acc[-1]:.2f}%")
    print(f"Final Val Accuracy: {val_acc[-1]:.2f}%")
    print(f"Final Train Loss: {train_loss[-1]:.4f}")
    print(f"Final Val Loss: {val_loss[-1]:.4f}")
    print("\nResults saved to:")
    print(f"  Figures: {exp_figures}")
    print(f"  Models: {exp_models}")
    print(f"  Metrics: {exp_metrics}")
    print("="*60)