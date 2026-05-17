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
from torch.utils.data import DataLoader, Dataset
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

def get_cifar_loader(root='./data/', batch_size=128, train=True, shuffle=True, num_workers=4, n_items=-1):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    if train:
        # 训练集数据增强
        data_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
            transforms.ToTensor(),  
            transforms.RandomRotation(15),  # 随机旋转±15度
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
            normalize  
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.ToTensor(),  
            normalize  
        ])

    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=data_transforms)
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  
    figures_path = os.path.join(project_root, 'reports', 'figures')
    os.makedirs(figures_path, exist_ok=True)

    train_loader = get_cifar_loader(batch_size=100, shuffle=True)
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