"""
Data loaders
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision.datasets.cifar")
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from torchvision import transforms
import torchvision.datasets as datasets

class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, index):  
        if index >= self.n_items:
            raise IndexError
        return self.dataset[index]

    def __len__(self):
        return min(self.n_items, len(self.dataset))

def get_cifar_loader(root='./data/', batch_size=128, train=True, shuffle=True, num_workers=4, 
                     n_items=-1, val_split=0.1, random_seed=42):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    # 训练集增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        normalize  
    ])

    # 验证集/测试集无增强
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),  
        normalize  
    ])

    # 自动下载逻辑
    dataset_dir = os.path.join(root, 'cifar-10-batches-py')
    required_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 
                      'data_batch_4', 'data_batch_5', 'test_batch', 'batches.meta']
    dataset_exists = os.path.exists(dataset_dir) and all(
        os.path.exists(os.path.join(dataset_dir, f)) for f in required_files
    )
    download = not dataset_exists

    if not train:
        dataset = datasets.CIFAR10(root=root, train=False, download=download, transform=val_test_transform)
        if n_items > 0:
            dataset = PartialDataset(dataset, n_items)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # 从官方50000张训练集中分出验证集
    # 先加载无transform的完整训练集
    full_train_dataset = datasets.CIFAR10(root=root, train=True, download=download, transform=None)
    
    # 计算划分大小（9:1 = 45000:5000）
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # 固定种子保证可复现划分
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    class TransformDataset(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __getitem__(self, index):
            img, label = self.dataset[index]
            return self.transform(img), label
        
        def __len__(self):
            return len(self.dataset)

    train_dataset = TransformDataset(train_dataset, train_transform)
    val_dataset = TransformDataset(val_dataset, val_test_transform)

    if n_items > 0:
        train_dataset = PartialDataset(train_dataset, n_items)
        val_dataset = PartialDataset(val_dataset, int(n_items * val_split))

    # 创建loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  
    figures_path = os.path.join(project_root, 'reports', 'figures')
    os.makedirs(figures_path, exist_ok=True)
    train_loader, _ = get_cifar_loader(batch_size=100, shuffle=True)
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

    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    fig.subplots_adjust(left=0.15, wspace=0.05, hspace=0.05)

    for cls_idx in range(10):
        for img_idx in range(10):
            ax = axes[cls_idx, img_idx]
            img = class_images[cls_idx][img_idx]
            img = img.permute(1, 2, 0).cpu().numpy()
            img = img * 0.5 + 0.5  
            ax.imshow(img)
            ax.axis('off')  

        axes[cls_idx, 0].text(
            -2.5, 16, cifar10_classes[cls_idx],
            ha='right', va='center', fontsize=11, weight='bold'
        )

    plt.savefig(os.path.join(figures_path, 'samples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 样本图已生成：reports/figures/samples.png")