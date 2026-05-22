import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from models.models import build_model

plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

ROOT = Path(__file__).parent.parent
REPORTS_DIR = os.path.join(ROOT, "reports")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["飞机", "汽车", "鸟类", "猫咪", "鹿", "狗狗", "青蛙", "马匹", "船只", "卡车"]

transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

inv_normalize = T.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, img_tensor, target_cls):
        self.model.eval()
        img_tensor.requires_grad = True
        
        output = self.model(img_tensor)
        loss = output[0, target_cls]
        
        self.model.zero_grad()
        loss.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = torch.relu(cam)
        
        cam = nn.functional.interpolate(
            cam.unsqueeze(0), 
            size=(32, 32), 
            mode="bilinear", 
            align_corners=False
        )
        
        cam -= cam.min()
        cam /= cam.max() + 1e-6
        
        return cam.squeeze().cpu().numpy()

def get_last_conv(model):
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return conv_layers[-1] if conv_layers else None

def save_gradcam_with_original(img_tensor, cam, label, save_path):
    # 转换为numpy格式
    img = inv_normalize(img_tensor[0]).permute(1,2,0).cpu().detach().numpy()
    img = np.clip(img, 0, 1)
    
    # 创建1行2列的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # 左边：原始图片
    ax1.imshow(img)
    ax1.set_title(f"原始图片 - {CLASS_NAMES[label]}")
    ax1.axis("off")
    
    # 右边：热力图叠加
    ax2.imshow(img)
    ax2.imshow(cam, cmap="jet", alpha=0.5)
    ax2.set_title(f"Grad-CAM 热力图")
    ax2.axis("off")
    
    # 调整布局
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def batch_gradcam():
    testset = torchvision.datasets.CIFAR10(
        root=os.path.join(ROOT, "data"), 
        train=False, 
        download=True, 
        transform=transform
    )
    samples = [12, 56, 89, 123, 456]

    for exp in os.listdir(REPORTS_DIR):
        exp_path = os.path.join(REPORTS_DIR, exp)
        if not os.path.isdir(exp_path):
            continue

        cfg_path = os.path.join(exp_path, "metrics", "config.txt")
        weight_path = os.path.join(exp_path, "models", "best_model.pth")
        if not os.path.exists(cfg_path) or not os.path.exists(weight_path):
            continue

        cfg = {}
        with open(cfg_path, encoding="utf-8") as f:
            for line in f:
                if ": " in line:
                    k, v = line.strip().split(": ", 1)
                    cfg[k] = v

        model_cfg = {
            "model_name": cfg.get("model", "original_residual_bn"),
            "activation": cfg.get("activation", "relu"),
            "conv_filters": int(cfg.get("conv_filters", 64)),
            "fc_neurons": int(cfg.get("fc_neurons", 512)),
            "num_classes": 10
        }
        if model_cfg["model_name"] == "original_bn_dropout":
            model_cfg["dropout_prob"] = float(cfg.get("dropout_prob", 0.0))

        try:
            model = build_model(model_cfg).to(DEVICE)
            weight_file = torch.load(weight_path, map_location=DEVICE, weights_only=False)

            if isinstance(weight_file, dict):
                if "state_dict" in weight_file:
                    state_dict = weight_file["state_dict"]
                elif "model" in weight_file:
                    state_dict = weight_file["model"]
                else:
                    state_dict = weight_file
            else:
                model = weight_file.to(DEVICE)
                model.eval()
                print(f"✅ {exp} 直接加载模型成功")
                layer = get_last_conv(model)
                if not layer:
                    continue
                cam = GradCAM(model, layer)
                for sid in samples:
                    img, label = testset[sid]
                    img = img.unsqueeze(0).to(DEVICE)
                    heatmap = cam.generate(img, label)
                    out_path = os.path.join(exp_path, "visualizations", "gradcam", f"sample_{sid}.png")
                    save_gradcam_with_original(img, heatmap, label, out_path)
                print(f"✅ {exp} Grad-CAM 已保存")
                continue

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            print(f"✅ {exp} 加载成功（自动处理键名）")

        except Exception as e:
            print(f"⚠️ {exp} 最终失败: {str(e)[:80]}")
            continue

        layer = get_last_conv(model)
        if not layer:
            continue

        cam = GradCAM(model, layer)
        for sid in samples:
            img, label = testset[sid]
            img = img.unsqueeze(0).to(DEVICE)
            heatmap = cam.generate(img, label)
            out_path = os.path.join(exp_path, "visualizations", "gradcam", f"sample_{sid}.png")
            save_gradcam_with_original(img, heatmap, label, out_path)

        print(f"✅ {exp} Grad-CAM 已保存")

if __name__ == "__main__":
    batch_gradcam()