import sys
import os
import time
import multiprocessing

multiprocessing.set_start_method('fork', force=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from data.loaders import get_cifar_loader
from utils.train_utils import get_accuracy, get_loss

EXP_ID = "35"
CONFIG_PATH = "configs/experiments.yaml"

if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ 检测到M系列芯片，使用Metal GPU加速")
    else:
        device = torch.device('cpu')
        print(f"⚠️  使用CPU运行，速度较慢")

    start_time = time.time()

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    exp_id = int(EXP_ID)
    exp_config = config['experiments'][exp_id]
    merged_args = config['defaults'].copy()
    merged_args.update(exp_config)

    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    args = Args(merged_args)

    exp_dir = os.path.join('reports', args.name)
    best_model_path = os.path.join(exp_dir, 'models', 'best_model.pth')
    config_path = os.path.join(exp_dir, 'metrics', 'config.txt')

    print(f"\n正在加载测试集...")
    test_loader = get_cifar_loader(
        train=False,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )
    print(f"测试集加载完成，耗时: {time.time() - start_time:.2f}秒")

    from models.models import MODEL_REGISTRY

    print("正在加载模型...")
    model_class = MODEL_REGISTRY[args.model]
    model_kwargs = {
        'activation': args.activation,
        'conv_filters': args.conv_filters,
        'fc_neurons': args.fc_neurons,
    }
    if args.model == "original_bn_dropout":
        model_kwargs["dropout_prob"] = args.dropout_prob

    model = model_class(**model_kwargs).to(device)
    
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")

    print("正在评估测试集...")
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    with torch.inference_mode():
        test_acc = get_accuracy(model, test_loader, device)
        test_loss = get_loss(model, test_loader, criterion, device)

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"实验名称：{args.name}")
    print(f"已加载最优模型 ✅")
    print(f"最终测试准确率：{test_acc:.2f}%")
    print(f"测试损失：{test_loss:.4f}")
    print(f"总耗时：{total_time:.2f}秒")
    print("=" * 60)

    with open(config_path, "a", encoding="utf-8") as f:
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n")

    print(f"✅ 已成功写入：{config_path}")