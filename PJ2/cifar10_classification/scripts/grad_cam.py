import os
import sys
import glob
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    def save_activation(self, module, inp, out):
        self.activations = out
    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]
    def __call__(self, x):
        self.model.eval()
        out = self.model(x)
        idx = torch.argmax(out, dim=1)
        self.model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot.scatter_(1, idx.unsqueeze(1), 1.0)
        out.backward(one_hot)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam = cam.detach().cpu().numpy()
        cam = np.array(Image.fromarray(cam).resize((32, 32)))
        return cam, idx.item()

def plot_grad_cam(model, loader, save_path):
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if not conv_layers:
        return
    target_layer = conv_layers[-1]
    cam_model = GradCAM(model, target_layer)
    x, y = next(iter(loader))
    device = next(model.parameters()).device
    x = x.to(device)
    cam, pred = cam_model(x)
    img = x[0].cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original\nPred: {pred}")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, type=str)
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
    from data.loaders import get_cifar_loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Original_BN_Dropout(
        activation=config.get('activation', 'relu'),
        conv_filters=int(config.get('conv_filters', 64)),
        fc_neurons=int(config.get('fc_neurons', 512)),
        dropout_prob=float(config.get('dropout_prob', 0.1))
    ).to(device)
    model_path = os.path.join(exp_dir, 'models', 'best_model.pth')
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    loader = get_cifar_loader(train=False, batch_size=1, num_workers=0)
    save_path = os.path.join(project_root, 'reports', 'figures', f'gradcam_exp_{args.exp}.png')
    plot_grad_cam(model, loader, save_path)
    print(f"✅ Grad-CAM 已保存：gradcam_exp_{args.exp}.png")