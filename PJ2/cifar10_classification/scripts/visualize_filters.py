import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn  
import matplotlib.pyplot as plt
import numpy as np
from models.models import build_model

plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent
REPORTS_DIR = os.path.join(ROOT, "reports")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def vis_filters(model, save_path, n=16):
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break
    if not first_conv:
        return
    w = first_conv.weight.detach().cpu().numpy()
    n = min(n, w.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(10,10))
    axes = axes.flatten()
    for i in range(n):
        img = np.mean(w[i], axis=0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
    plt.suptitle("第一层卷积核可视化", fontsize=14, y=0.95)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
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
                out_path = os.path.join(exp_path, "visualizations", "filters", "first_layer.png")
                vis_filters(model, out_path)
                print(f"✅ {exp} 卷积核已保存")
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

        out_path = os.path.join(exp_path, "visualizations", "filters", "first_layer.png")
        vis_filters(model, out_path)
        print(f"✅ {exp} 卷积核已保存")