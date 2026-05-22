"""
任务二：数据加载器
"""
import os
import sys
import warnings
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms

# 过滤警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision.datasets.cifar")

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 全局缓存：确保整个程序生命周期内划分固定
_FULL_TRAIN_SET = None
_TRAIN_SET = None
_VAL_SET = None

class PartialDataset(Dataset):
    """子集数据集"""
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = min(n_items, len(dataset))

    def __getitem__(self, index):
        if index >= self.n_items:
            raise IndexError(f"索引 {index} 超出子集大小 {self.n_items}")
        return self.dataset[index]

    def __len__(self):
        return self.n_items

def get_cifar_loader(root='./data/', batch_size=128, train=True, val=False, shuffle=True, num_workers=4, n_items=-1):
    global _FULL_TRAIN_SET, _TRAIN_SET, _VAL_SET

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    data_transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
    ])

    data_transforms_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if _FULL_TRAIN_SET is None:
        print("📥 首次加载CIFAR10数据集（仅下载一次）")
        _FULL_TRAIN_SET = datasets.CIFAR10(
            root=root, train=True, download=True, transform=None
        )
        
        generator = torch.Generator().manual_seed(2020)
        train_subset, val_subset = random_split(
            range(len(_FULL_TRAIN_SET)), [45000, 5000], generator=generator
        )

        train_dataset = datasets.CIFAR10(
            root=root, train=True, download=False, transform=data_transforms_train
        )
        _TRAIN_SET = Subset(train_dataset, train_subset.indices)

        val_dataset = datasets.CIFAR10(
            root=root, train=True, download=False, transform=data_transforms_test
        )
        _VAL_SET = Subset(val_dataset, val_subset.indices)

    # ✅ 唯一正确的逻辑，没有任何歧义
    if val:
        dataset = _VAL_SET
    elif train:
        dataset = _TRAIN_SET
    else:
        dataset = datasets.CIFAR10(
            root=root, train=False, download=True, transform=data_transforms_test
        )

    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=train and not val,
        persistent_workers=True if num_workers > 0 else False
    )
    return loader


if __name__ == '__main__':
    # 生成样本图
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    figures_path = os.path.join(project_root, 'reports', 'figures')
    os.makedirs(figures_path, exist_ok=True)

    train_loader = get_cifar_loader(batch_size=100, shuffle=True, num_workers=0)
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    samples_per_class = 10
    class_images = {i: [] for i in range(10)}
    for imgs, labels in train_loader:
        for img, label in zip(imgs, labels):
            l = label.item()
            if len(class_images[l]) < samples_per_class:
                class_images[l].append(img)
        if all(len(v) == samples_per_class for v in class_images.values()):
            break

    # 绘制样本图
    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    fig.subplots_adjust(left=0.15, wspace=0.05, hspace=0.05)

    for cls_idx in range(10):
        for img_idx in range(10):
            ax = axes[cls_idx, img_idx]
            img = class_images[cls_idx][img_idx]
            img = img.permute(1, 2, 0).cpu().numpy()
            img = img * 0.5 + 0.5  # 反归一化
            ax.imshow(img)
            ax.axis('off')

        # 添加类别标签
        axes[cls_idx, 0].text(
            -2.5, 16, cifar10_classes[cls_idx],
            ha='right', va='center', fontsize=11, weight='bold'
        )

    plt.savefig(os.path.join(figures_path, 'samples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 样本图已生成：reports/figures/samples.png")