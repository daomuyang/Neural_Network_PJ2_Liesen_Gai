import sys
import os
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)  # 优先导入项目根目录

warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='numpy')
warnings.filterwarnings('ignore', category=FutureWarning, module='matplotlib')

import matplotlib as mpl
mpl.use('Agg')  
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import argparse
import yaml
from tqdm import tqdm

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader
from utils.train_utils import (
    get_accuracy, get_loss, set_random_seeds,
    create_experiment_dirs, get_middle_conv_layer
)

# 模型注册表
MODEL_REGISTRY = {
    'VGG_A': VGG_A,
    'VGG_A_BatchNorm': VGG_A_BatchNorm
}

def train(model, optimizer, criterion, train_loader, val_loader, device,
          exp_figures, exp_models, epochs_n=100, scheduler=None):
    """核心训练函数"""
    model.to(device)
    # 初始化曲线数组
    train_loss_curve = np.zeros(epochs_n, dtype=np.float32)
    val_loss_curve = np.zeros(epochs_n, dtype=np.float32)
    train_acc_curve = np.zeros(epochs_n, dtype=np.float32)
    val_acc_curve = np.zeros(epochs_n, dtype=np.float32)
    max_val_acc = 0.0
    best_model_path = os.path.join(exp_models, 'best_model.pth')
    losses_list = []
    grad_last_layer_avg = np.zeros(epochs_n, dtype=np.float32)
    grad_mid_layer_avg = np.zeros(epochs_n, dtype=np.float32)
    batches_n = len(train_loader)
    
    # 校验中间卷积层
    try:
        mid_conv_layer = get_middle_conv_layer(model)
        print(f"✅ 找到中间卷积层: {mid_conv_layer}")
    except Exception as e:
        raise RuntimeError(f"获取中间卷积层失败: {e}")

    # GPU预热
    print("🔄 正在预热GPU...")
    with torch.no_grad():
        dummy_x = torch.randn(1, 3, 32, 32, device=device, dtype=torch.float32)
        model(dummy_x)
    torch.cuda.empty_cache()  # 清理预热显存
    print("✅ GPU预热完成，开始训练")

    # 训练主循环
    for epoch in tqdm(range(epochs_n), unit='epoch', desc=f'Training {model.__class__.__name__}'):
        model.train()
        epoch_train_loss = 0.0
        step_losses = []
        epoch_grad_last_sum = 0.0
        epoch_grad_mid_sum = 0.0

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True, dtype=torch.float32)
            y = y.to(device, non_blocking=True, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            preds = model(x)
            loss = criterion(preds, y)

            # 反向传播
            loss.backward()
            
            # 梯度范数计算
            try:
                last_layer_grad = model.classifier[-1].weight.grad
                grad_last_norm = torch.norm(last_layer_grad).item() if last_layer_grad is not None else 0.0
                epoch_grad_last_sum += grad_last_norm

                mid_layer_grad = mid_conv_layer.weight.grad
                grad_mid_norm = torch.norm(mid_layer_grad).item() if mid_layer_grad is not None else 0.0
                epoch_grad_mid_sum += grad_mid_norm
            except Exception as e:
                print(f"⚠️ 第{epoch+1}轮第{step+1}步梯度计算失败: {e}")
                grad_last_norm = 0.0
                grad_mid_norm = 0.0

            optimizer.step()
            epoch_train_loss += loss.item()
            step_losses.append(loss.item())

        # 学习率调度
        if scheduler is not None:
            scheduler.step()

        # 计算本轮指标
        with torch.no_grad():
            train_loss_curve[epoch] = epoch_train_loss / batches_n if batches_n > 0 else 0.0
            val_loss_curve[epoch] = get_loss(model, val_loader, criterion, device)
            train_acc_curve[epoch] = get_accuracy(model, train_loader, device)
            val_acc_curve[epoch] = get_accuracy(model, val_loader, device)

        grad_last_layer_avg[epoch] = epoch_grad_last_sum / batches_n if batches_n > 0 else 0.0
        grad_mid_layer_avg[epoch] = epoch_grad_mid_sum / batches_n if batches_n > 0 else 0.0

        # 保存最优模型
        current_val_acc = val_acc_curve[epoch]
        if current_val_acc > max_val_acc:
            max_val_acc = current_val_acc
            # 先保存临时文件，再重命名
            temp_path = best_model_path + '.tmp'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': max_val_acc,
                'val_loss': val_loss_curve[epoch],
            }, temp_path)
            os.rename(temp_path, best_model_path)
            tqdm.write(f"✅ Epoch {epoch+1:2d} | Val Acc: {max_val_acc:.2f}% | Val Loss: {val_loss_curve[epoch]:.4f} | NEW BEST")
        else:
            tqdm.write(f"Epoch {epoch+1:2d} | Val Acc: {current_val_acc:.2f}% | Val Loss: {val_loss_curve[epoch]:.4f}")

        losses_list.append(step_losses)
        torch.cuda.empty_cache()  # 每轮清理显存

    # 绘制训练曲线
    print("🔄 正在绘制训练曲线...")
    try:
        plt.figure(figsize=(12, 4), dpi=100)
        plt.subplot(1,2,1)
        plt.plot(train_loss_curve, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
        plt.plot(val_loss_curve, 'orange', label='Val Loss', linewidth=2, alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1,2,2)
        plt.plot(train_acc_curve, 'r-', label='Train Acc', linewidth=2, alpha=0.8)
        plt.plot(val_acc_curve, 'g-', label='Val Acc', linewidth=2, alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_figures, 'training_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 训练曲线绘制完成")
    except Exception as e:
        print(f"⚠️ 绘制曲线失败: {e}")

    return losses_list, grad_last_layer_avg, grad_mid_layer_avg, train_loss_curve, val_loss_curve, train_acc_curve, val_acc_curve, max_val_acc

def evaluate_test(model, test_loader, device):
    """在测试集上评估模型"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True, dtype=torch.float32)
            y = y.to(device, non_blocking=True, dtype=torch.long)
            pred = model(x)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
    
    test_acc = 100 * correct / total if total > 0 else 0.0
    test_error = 100 - test_acc
    return test_acc, test_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGG BatchNorm Loss Landscape Experiment')
    parser.add_argument('--exp', type=str, required=True, help='实验ID（对应experiments.yaml）')
    parser.add_argument('--config', type=str, default='configs/experiments.yaml', help='配置文件路径')
    parser.add_argument('--model', type=str, choices=MODEL_REGISTRY.keys(), help='覆盖模型类型')
    parser.add_argument('--experiment', type=str, help='覆盖实验名称')
    parser.add_argument('--batch_size', type=int, help='覆盖批次大小')
    parser.add_argument('--epochs', type=int, help='覆盖训练轮数')
    parser.add_argument('--lr', type=float, help='覆盖学习率')
    parser.add_argument('--seed', type=int, help='覆盖随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数（Windows建议0）')

    args = parser.parse_args()

    # 加载配置文件
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败: {e}")

    # 合并配置
    exp_id = int(args.exp)
    if exp_id not in config['experiments']:
        raise ValueError(f"实验ID {exp_id} 不存在于配置文件")
    
    exp_config = config['experiments'][exp_id]
    merged_args = config['defaults'].copy()
    merged_args.update(exp_config)

    # 覆盖命令行参数
    for k, v in vars(args).items():
        if v is not None and k not in ['exp', 'config']:
            if k == 'experiment':
                merged_args['name'] = v
            else:
                merged_args[k] = v

    # 转换为Namespace
    args = argparse.Namespace(**merged_args)

    # 设备选择
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        print("✅ 使用GPU训练")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ 使用MPS训练")
    else:
        device = torch.device('cpu')
        print("⚠️ 使用CPU训练")

    print("="*60)
    print(f"Experiment ID: {exp_id} | Model: {args.model} | LR: {args.lr}")
    print("="*60)

    # 创建实验目录
    exp_figures, exp_models, exp_metrics = create_experiment_dirs(args.name)
    # 固定随机种子
    set_random_seeds(int(args.seed), device)

    # 加载数据
    print("🔄 正在加载数据...")
    try:
        train_loader = get_cifar_loader(
            train=True, 
            batch_size=int(args.batch_size), 
            num_workers=int(args.num_workers)
        )
        val_loader = get_cifar_loader(
            val=True, 
            shuffle=False, 
            batch_size=int(args.batch_size), 
            num_workers=int(args.num_workers)
        )
        test_loader = get_cifar_loader(
            train=False, 
            batch_size=int(args.batch_size), 
            num_workers=int(args.num_workers)
        )
        print("✅ 数据加载完成")
    except Exception as e:
        raise RuntimeError(f"数据加载失败: {e}")

    # 初始化模型和优化器
    model = MODEL_REGISTRY[args.model]()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=float(args.lr), 
        weight_decay=float(args.weight_decay)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    
    # 学习率调度器
    scheduler = None
    if hasattr(args, 'scheduler') and args.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=int(args.T_max)
        )

    # 开始训练
    try:
        losses_list, grad_last_layer_avg, grad_mid_layer_avg, train_loss, val_loss, train_acc, val_acc, best_acc = train(
            model, optimizer, criterion, train_loader, val_loader, device,
            exp_figures, exp_models, epochs_n=int(args.epochs), scheduler=scheduler
        )
    except Exception as e:
        raise RuntimeError(f"训练过程失败: {e}")

    # 加载最优模型评估测试集
    print("\n" + "="*60)
    print("[Final Test] 加载最优模型...")
    best_model_path = os.path.join(exp_models, 'best_model.pth')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"最优模型文件不存在: {best_model_path}")
    
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    
    test_acc, test_error = evaluate_test(model, test_loader, device)
    print(f"✅ 最终测试准确率: {test_acc:.2f}%")
    print(f"✅ 最终测试误差: {test_error:.2f}%")
    print("="*60)

    # 保存所有指标
    print("\n正在保存数据...")
    try:
        # 展平损失列表
        flat_losses = [loss for epoch_loss in losses_list for loss in epoch_loss]
        np.savetxt(os.path.join(exp_metrics, 'losses.txt'), flat_losses, fmt='%.6f')
        # 保存梯度数据
        np.save(os.path.join(exp_metrics, 'grad_last_layer_avg.npy'), grad_last_layer_avg)
        np.save(os.path.join(exp_metrics, 'grad_mid_layer_avg.npy'), grad_mid_layer_avg)
        # 保存训练曲线
        np.savez(
            os.path.join(exp_metrics, 'training_curves.npz'),
            train_loss=train_loss, 
            val_loss=val_loss, 
            train_acc=train_acc, 
            val_acc=val_acc
        )

        # 保存配置文件
        exp_root = os.path.dirname(exp_figures)
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
            f.write(f"Scheduler: {args.scheduler if hasattr(args, 'scheduler') else 'None'}\n")
            f.write(f"T_max: {args.T_max if hasattr(args, 'T_max') else 'None'}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Num Workers: {args.num_workers}\n")
            f.write(f"Loss Type: {args.loss_type}\n")
            f.write(f"Note: {args.note if hasattr(args, 'note') else 'None'}\n")
            f.write("-"*60 + "\n")
            f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n")
            f.write(f"Final Test Accuracy: {test_acc:.2f}%\n")
            f.write(f"Final Test Error: {test_error:.2f}%\n")
        print("✅ 所有数据保存完成")
    except Exception as e:
        raise RuntimeError(f"保存数据失败: {e}")

    # 最终结果输出
    print("\n🎉 训练完成！")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"最终测试准确率: {test_acc:.2f}%")
    print(f"最终测试误差: {test_error:.2f}%")
    print("="*60)