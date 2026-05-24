import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_metrics(exp_name):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(project_root, 'reports', exp_name, 'metrics')
    return (
        np.load(os.path.join(metrics_path, 'grad_predictiveness.npy')),
        np.load(os.path.join(metrics_path, 'max_grad_diff.npy'))
    )

def moving_average(data, window=200):
    return np.convolve(data, np.ones(window)/window, mode='valid')

if __name__ == '__main__':
    exp_no_bn = 'vgg_a_lr001'
    exp_bn = 'vgg_a_bn_lr001'
    
    pred_no_bn, beta_no_bn = load_metrics(exp_no_bn)
    pred_bn, beta_bn = load_metrics(exp_bn)
    
    steps = np.arange(len(moving_average(pred_no_bn)))
    
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports', 'figures')
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(steps, moving_average(beta_no_bn), 'r-', linewidth=2, label='Standard VGG', alpha=0.8)
    plt.plot(steps, moving_average(beta_bn), 'b-', linewidth=2, label='Standard VGG + BatchNorm', alpha=0.8)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Effective β-smoothness', fontsize=12)
    plt.title('Maximum Gradient Difference Over Distance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'paper_beta_smoothness.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6), dpi=300)
    grad_error_no_bn = 1 - np.clip(pred_no_bn, 0, 1)
    grad_error_bn = 1 - np.clip(pred_bn, 0, 1)
    plt.plot(steps, moving_average(grad_error_no_bn), 'r-', linewidth=2, label='Standard VGG', alpha=0.8)
    plt.plot(steps, moving_average(grad_error_bn), 'b-', linewidth=2, label='Standard VGG + BatchNorm', alpha=0.8)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Gradient L2 Difference', fontsize=12)
    plt.title('Gradient Predictiveness', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'paper_gradient_predictiveness.png'), bbox_inches='tight')
    plt.close()

    print("="*60)
    print("📊 统计结果")
    print("="*60)
    print(f"无BN 平均有效β平滑度: {np.mean(beta_no_bn):.2f}")
    print(f"有BN 平均有效β平滑度: {np.mean(beta_bn):.2f}")
    print(f"BN降低β平滑度: {(1 - np.mean(beta_bn)/np.mean(beta_no_bn))*100:.2f}%\n")
    print(f"无BN 平均梯度相对误差: {np.mean(grad_error_no_bn):.4f}")
    print(f"有BN 平均梯度相对误差: {np.mean(grad_error_bn):.4f}")
    print(f"BN降低梯度误差: {(1 - np.mean(grad_error_bn)/np.mean(grad_error_no_bn))*100:.2f}%")
    print("="*60)