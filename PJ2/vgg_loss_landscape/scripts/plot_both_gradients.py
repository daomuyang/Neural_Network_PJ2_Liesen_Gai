import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_grad(exp_name, layer_type):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if layer_type == 'last':
        path = os.path.join(root, 'reports', exp_name, 'metrics', 'grad_last_layer.npy')
        if not os.path.exists(path):
            path = os.path.join(root, 'reports', exp_name, 'metrics', 'grad_last_layer_avg.npy')
    else:
        path = os.path.join(root, 'reports', exp_name, 'metrics', 'grad_mid_layer.npy')
        if not os.path.exists(path):
            path = os.path.join(root, 'reports', exp_name, 'metrics', 'grad_mid_layer_avg.npy')
    return np.load(path)

def plot(layer_type, save_path):
    a = load_grad('vgg_a_lr001', layer_type)
    bn = load_grad('vgg_a_bn_lr001', layer_type)

    plt.figure(figsize=(10,6))
    plt.plot(np.arange(1,31), a, 'r-', linewidth=2.5, label='VGG-A (no BN)')
    plt.plot(np.arange(1,31), bn, 'b-', linewidth=2.5, label='VGG-A (with BN)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Gradient Norm')
    if layer_type == 'last':
        plt.title('Last Layer Gradient Norm Comparison')
    else:
        plt.title('Middle Convolutional Layer Gradient Norm Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, 'reports/figures')
    os.makedirs(out, exist_ok=True)

    plot('last', os.path.join(out, 'gradient_norm_last_layer.png'))
    plot('mid', os.path.join(out, 'gradient_norm_mid_layer.png'))

    a_last = load_grad('vgg_a_lr001', 'last')
    bn_last = load_grad('vgg_a_bn_lr001', 'last')
    a_mid = load_grad('vgg_a_lr001', 'mid')
    bn_mid = load_grad('vgg_a_bn_lr001', 'mid')

    max_last_a = np.max(a_last)
    max_last_bn = np.max(bn_last)
    mean_mid_a = np.mean(a_mid[15:])
    mean_mid_bn = np.mean(bn_mid[15:])

    report = os.path.join(root, 'reports', 'gradient_result.txt')
    with open(report, 'w', encoding='utf-8') as f:
        f.write("Gradient Norm Experimental Results\n")
        f.write("-"*40 + "\n")
        f.write(f"Last Layer Max Norm:\n")
        f.write(f"VGG-A        {max_last_a:.4f}\n")
        f.write(f"VGG-A-BN     {max_last_bn:.4f}\n\n")
        f.write(f"Middle Layer Steady Mean (epoch15-30):\n")
        f.write(f"VGG-A        {mean_mid_a:.4f}\n")
        f.write(f"VGG-A-BN     {mean_mid_bn:.4f}\n")

    print("🎉 两张图生成完成！")
    print("📄 gradient_result.txt 已自动生成")