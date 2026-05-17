import os
import sys
import glob
import argparse
import torch
import matplotlib.pyplot as plt
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def visualize_filters(model, layer_idx=0, save_path=None):
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if not conv_layers:
        return
    layer = conv_layers[layer_idx]
    weights = layer.weight.data.cpu().numpy()
    n = min(16, weights.shape[0])
    plt.figure(figsize=(12, 8))
    for i in range(n):
        plt.subplot(2, 8, i + 1)
        img = weights[i].mean(axis=0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, type=str)
    parser.add_argument('--layer', default=0, type=int)
    args = parser.parse_args()
    exp_dirs = glob.glob(os.path.join(project_root, 'reports', f'{args.exp}_*'))
    if not exp_dirs:
        print("❌ 实验文件夹不存在")
        exit(1)
    exp_dir = exp_dirs[0]
    config_path = os.path.join(exp_dir, 'metrics', 'config.txt')
    config = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ': ' in line:
                k, v = line.split(': ', 1)
                config[k] = v

    from models.models import Original_BN_Dropout
    model = Original_BN_Dropout(
        activation=config.get('activation', 'relu'),
        conv_filters=int(config.get('conv_filters', 64)),
        fc_neurons=int(config.get('fc_neurons', 512)),
        dropout_prob=float(config.get('dropout_prob', 0.1))
    )
    model_path = os.path.join(exp_dir, 'models', 'best_model.pth')
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    save_path = os.path.join(project_root, 'reports', 'figures', f'filters_exp_{args.exp}.png')
    visualize_filters(model, args.layer, save_path)
    print(f"✅ 卷积核已保存：filters_exp_{args.exp}.png")