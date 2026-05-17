import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_losses(experiment_name):
    """加载指定实验的step级loss数据"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loss_path = os.path.join(project_root, 'reports', experiment_name, 'metrics', 'losses.txt')
    return np.loadtxt(loss_path)

def load_training_curves(experiment_name):
    """加载指定实验的epoch级训练曲线数据"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    curves_path = os.path.join(project_root, 'reports', experiment_name, 'metrics', 'training_curves.npz')
    data = np.load(curves_path)

    if 'loss' in data:
        train_loss = data['loss']
        val_loss = None
    else:
        train_loss = data['train_loss']
        val_loss = data['val_loss']
    
    return train_loss, val_loss, data['train_acc'], data['val_acc']

def compute_model_loss_bounds(all_losses):
    """
    计算单个模型的损失边界：对每个step，取该模型所有学习率下loss的最大值和最小值
    """
    min_length = min(len(loss) for loss in all_losses)
    aligned_losses = np.array([loss[:min_length] for loss in all_losses])
    
    min_bound = np.min(aligned_losses, axis=0)
    max_bound = np.max(aligned_losses, axis=0)
    
    return min_bound, max_bound

def plot_training_curves_comparison(save_path):
    """
    相同配置下VGG-A与VGG-A-BN的训练曲线对比
    """
    vgg_a_train_loss, vgg_a_val_loss, vgg_a_train_acc, vgg_a_val_acc = load_training_curves('vgg_a_lr001')
    vgg_a_bn_train_loss, vgg_a_bn_val_loss, vgg_a_bn_train_acc, vgg_a_bn_val_acc = load_training_curves('vgg_a_bn_lr001')
    
    epochs = np.arange(1, len(vgg_a_train_loss)+1)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, vgg_a_train_loss, 'b-', label='VGG-A Train Loss', linewidth=2, alpha=0.8)
    if vgg_a_val_loss is not None:
        plt.plot(epochs, vgg_a_val_loss, 'b--', label='VGG-A Val Loss', linewidth=2, alpha=0.8)
    plt.plot(epochs, vgg_a_bn_train_loss, 'r-', label='VGG-A-BN Train Loss', linewidth=2, alpha=0.8)
    if vgg_a_bn_val_loss is not None:
        plt.plot(epochs, vgg_a_bn_val_loss, 'r--', label='VGG-A-BN Val Loss', linewidth=2, alpha=0.8)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Loss Curve Comparison (lr=0.001)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs[::2])
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, vgg_a_val_acc, 'b-', label='VGG-A Val Acc', linewidth=2, alpha=0.8)
    plt.plot(epochs, vgg_a_bn_val_acc, 'r-', label='VGG-A-BN Val Acc', linewidth=2, alpha=0.8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('Validation Accuracy Comparison (lr=0.001)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs[::2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 训练曲线对比图已保存到: {save_path}")

def plot_standard_loss_landscape(vgg_a_bounds, vgg_a_bn_bounds, save_path):
    """
    绘制标准损失景观主图，展示VGG-A与VGG-A-BN在不同学习率下的损失波动范围（以阴影区域表示）
    """
    steps = np.arange(len(vgg_a_bounds[0]))
    
    plt.figure(figsize=(10, 6)) 
    
    plt.plot(steps, vgg_a_bounds[0], color='#888888', linewidth=1.5, linestyle='--')
    plt.plot(steps, vgg_a_bounds[1], color='#888888', linewidth=1.5)
    plt.fill_between(
        steps, vgg_a_bounds[0], vgg_a_bounds[1],
        color='#888888', alpha=0.3, label='Standard VGG'
    )
    
    plt.plot(steps, vgg_a_bn_bounds[0], color='#1f77b4', linewidth=1.5, linestyle='--')
    plt.plot(steps, vgg_a_bn_bounds[1], color='#1f77b4', linewidth=1.5)
    plt.fill_between(
        steps, vgg_a_bn_bounds[0], vgg_a_bn_bounds[1],
        color='#1f77b4', alpha=0.3, label='Standard VGG + BatchNorm'
    )
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Loss Landscape', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='-')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 标准损失景观主图已保存到: {save_path}")
    print(f"📊 无BN模型损失波动范围: {vgg_a_bounds[0].min():.2f} ~ {vgg_a_bounds[1].max():.2f}")
    print(f"📊 有BN模型损失波动范围: {vgg_a_bn_bounds[0].min():.2f} ~ {vgg_a_bn_bounds[1].max():.2f}")

def plot_learning_rate_effect(save_path):
    """
    不同学习率下的损失曲线对比
    """
    plt.figure(figsize=(12, 8))
    
    lrs = ['2e-3', '1e-3', '5e-4', '1e-4']
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    vgg_a_exps = ['vgg_a_lr002', 'vgg_a_lr001', 'vgg_a_lr0005', 'vgg_a_lr0001']
    vgg_a_bn_exps = ['vgg_a_bn_lr002', 'vgg_a_bn_lr001', 'vgg_a_bn_lr0005', 'vgg_a_bn_lr0001']
    
    plt.subplot(2, 1, 1)
    for i, exp_name in enumerate(vgg_a_exps):
        loss = load_losses(exp_name)
        plt.plot(loss, color=colors[i], label=f'lr={lrs[i]}', alpha=0.7)
    plt.title('VGG-A (without BatchNorm) Loss Curves at Different Learning Rates', fontsize=12, fontweight='bold')
    plt.ylabel('Cross-Entropy Loss', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 3) 

    plt.subplot(2, 1, 2)
    for i, exp_name in enumerate(vgg_a_bn_exps):
        loss = load_losses(exp_name)
        plt.plot(loss, color=colors[i], label=f'lr={lrs[i]}', alpha=0.7)
    plt.title('VGG-A (with BatchNorm) Loss Curves at Different Learning Rates', fontsize=12, fontweight='bold')
    plt.xlabel('Training Step', fontsize=10)
    plt.ylabel('Cross-Entropy Loss', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 3)  
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 学习率影响对比子图已保存到: {save_path}")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(project_root, 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    vgg_a_all_losses = [
        load_losses('vgg_a_lr002'),    # 2e-3
        load_losses('vgg_a_lr001'),    # 1e-3
        load_losses('vgg_a_lr0005'),   # 5e-4
        load_losses('vgg_a_lr0001')    # 1e-4
    ]
    
    vgg_a_bn_all_losses = [
        load_losses('vgg_a_bn_lr002'),    # 2e-3
        load_losses('vgg_a_bn_lr001'),    # 1e-3
        load_losses('vgg_a_bn_lr0005'),   # 5e-4
        load_losses('vgg_a_bn_lr0001')    # 1e-4
    ]

    vgg_a_bounds = compute_model_loss_bounds(vgg_a_all_losses)
    vgg_a_bn_bounds = compute_model_loss_bounds(vgg_a_bn_all_losses)
    
    # 生成图表，全部保存到reports/figures/
    # 训练曲线对比图
    training_fig_path = os.path.join(figures_dir, 'training_curves_comparison.png')
    plot_training_curves_comparison(training_fig_path)
    
    # 标准损失景观主图
    main_fig_path = os.path.join(figures_dir, 'final_loss_landscape.png')
    plot_standard_loss_landscape(vgg_a_bounds, vgg_a_bn_bounds, main_fig_path)
    
    # 学习率对比子图
    extra_fig_path = os.path.join(figures_dir, 'learning_rate_effect.png')
    plot_learning_rate_effect(extra_fig_path)
    
    print("\n🎉 所有图表生成完成！全部保存到 reports/figures/ 目录下")
    print(f"📌 训练曲线对比 - {training_fig_path}")
    print(f"📌 损失景观对比 - {main_fig_path}")
    print(f"📌 学习率影响对比 - {extra_fig_path}")