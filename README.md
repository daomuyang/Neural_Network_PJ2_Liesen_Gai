# Neural Network and Deep Learning Project 2
## 盖烈森 23307130013@m.fudan.edu.cn
CIFAR-10图像分类与Batch Normalization机制研究

## 项目简介
本项目是神经网络与深度学习课程的第二个课程项目，包含两个独立的实验任务：
1. **任务一**：在CIFAR-10数据集上训练卷积神经网络，通过多种优化手段提升分类性能
2. **任务二**：深入研究Batch Normalization的作用机制，通过控制变量实验验证其对损失景观和梯度传播的影响

所有实验均基于PyTorch框架实现，代码结构清晰可复现。

## GitHub项目结构
```
PJ2/
├── cifar10_classification/    # 任务一：CIFAR-10图像分类实验根目录
│   ├── utils/                  # 工具函数目录
│   │   ├── __init__.py         # 包初始化文件
│   │   ├── train_utils.py      # 训练通用工具函数
│   │   └── custom_optimizers.py # 自定义优化器实现
│   ├── models/                 # 模型定义目录
│   │   ├── __init__.py         # 包初始化文件
│   │   └── models.py           # 任务一所有网络模型定义
│   ├── configs/                # 实验配置目录
│   │   └── experiments.yaml    # 实验超参数配置文件
│   ├── scripts/                # 可执行脚本目录
│   │   ├── train.py            # 训练主脚本
│   │   ├── summarize_all.py    # 实验结果汇总脚本
│   │   ├── grad_cam.py         # Grad-CAM可视化脚本
│   │   └── visualize_filters.py # 卷积核可视化脚本
│   ├── data/                   # 数据处理目录
│   │   ├── __init__.py         # 包初始化文件
│   │   └── loaders.py          # 数据集加载与预处理
│   └── reports/                # 实验结果输出目录
│       ├── all_experiments_summary.md # 全部实验汇总文档
│       ├── figures/            # 汇总图表保存目录
│       └── [实验名]/           # 单个实验结果文件夹
│           ├── config.txt      # 实验配置记录文件
│           ├── metrics/        # 实验指标目录
│           │   ├── losses.txt  # 训练损失记录
│           │   ├── config.txt  # 实验配置、准确率、参数量记录
│           │   └── training_curves.npz # 训练曲线数据
│           └── figures/        # 实验图表目录
│               └── training_curve.png # 训练损失与准确率曲线
├── vgg_loss_landscape/        # 任务二：BN与损失景观实验根目录
│   ├── utils/                  # 工具函数目录
│   │   ├── __init__.py         # 包初始化文件
│   │   └── train_utils.py      # 训练通用工具函数
│   ├── models/                 # 模型定义目录
│   │   ├── __init__.py         # 包初始化文件
│   │   ├── vgg.py              # VGG-A与VGG-A-BN模型
│   │   └── nn.py               # 基础网络组件
│   ├── configs/                # 实验配置目录
│   │   └── experiments.yaml    # 实验超参数配置文件
│   ├── scripts/                # 可执行脚本目录
│   │   ├── train.py            # 训练主脚本
│   │   ├── summarize_results.py # 实验结果汇总
│   │   ├── plot_both_gradients.py # 梯度范数对比绘图
│   │   └── plot_combined_loss_landscape.py # 损失景观对比绘图
│   ├── data/                   # 数据处理目录
│   │   ├── __init__.py         # 包初始化文件
│   │   └── loaders.py          # 数据集加载与预处理
│   └── reports/                # 实验结果输出目录
│       ├── experiment_results.txt # 实验结果汇总文本
│       ├── gradient_result.txt # 梯度实验结果文本
│       ├── figures/            # 汇总图表保存目录
│       └── [实验名]/           # 单个实验结果文件夹
│           ├── config.txt      # 实验配置记录文件
│           ├── metrics/        # 实验指标目录
│           │   ├── losses.txt  # 训练损失记录
│           │   ├── training_curves.npz # 训练曲线数据
│           │   ├── grad_last_layer_avg.npy # 最后一层梯度数据
│           │   └── grad_mid_layer_avg.npy # 中间层梯度数据
│           └── figures/        # 实验图表目录
│               └── training_curve.png # 训练损失与准确率曲线
└── 论文/                     # 论文资料
    ├── bib.bib                 # LaTeX参考文献库
    ├── Gaussian Error Linear Units.pdf # GELU激活函数论文
    ├── How does Batch Normalization help optimization.pdf # BN优化机制论文
    ├── Deep_Residual_Learning_for_Image_Recognition.pdf # 残差网络ResNet论文
    ├── Grad-CAM_Visual_Explanations_from_Deep_Networks_via_Gradient-Based_Localization.pdf # Grad-CAM可解释性论文
    └── Decoupled Weight Decay Regularization.pdf # 解耦权重衰减正则化论文
```

> **说明**：
> - `__pycache__`、`.DS_Store`等系统缓存文件未上传
> - 数据集文件和训练好的模型权重文件未上传，具体下载方式见文末重要说明
> - 所有`data/`目录仅包含数据加载代码，不包含数据集文件

## 环境配置
### 基础环境
- 操作系统：Ubuntu 22.04 LTS / macOS 15.5
- Python版本：3.11 / 3.13
- CUDA版本：12.8.1
- PyTorch版本：2.9.1

### 依赖安装
```bash
pip install torch==2.9.1 torchvision==0.20.1
pip install numpy==1.26.4 matplotlib==3.9.0 pyyaml==6.0.1 tqdm==4.66.4 pillow==10.3.0
```

### 硬件要求
- 推荐配置：NVIDIA GPU（显存≥8GB）

## 运行指南
### 任务一：CIFAR-10分类实验
**所有命令均在`cifar10_classification`目录下执行**

#### 1. 训练实验
```bash
# 运行单个实验（--exp指定实验编号）
python3 scripts/train.py --exp 00  # 原始基线模型
python3 scripts/train.py --exp 01  # BN模型
python3 scripts/train.py --exp 02  # BN+Dropout模型
python3 scripts/train.py --exp 12  # 最优组合模型
```

#### 2. 结果可视化
```bash
# 生成所有实验结果汇总表
python3 scripts/summarize_all.py

# 可视化指定实验的卷积滤波器（--exp指定实验编号）
python3 scripts/visualize_filters.py --exp 12

# 生成指定实验的Grad-CAM热力图（--exp指定实验编号）
python3 scripts/grad_cam.py --exp 12
```

### 任务二：Batch Normalization研究
**所有命令均在`vgg_loss_landscape`目录下执行**

#### 1. 训练实验
```bash
# 运行单个实验（--exp指定实验编号）
python3 scripts/train.py --exp 00  # VGG-A lr=1e-4
python3 scripts/train.py --exp 01  # VGG-A lr=5e-4
python3 scripts/train.py --exp 02  # VGG-A lr=1e-3
python3 scripts/train.py --exp 03  # VGG-A lr=2e-3
python3 scripts/train.py --exp 04  # VGG-A-BN lr=1e-4
python3 scripts/train.py --exp 05  # VGG-A-BN lr=5e-4
python3 scripts/train.py --exp 06  # VGG-A-BN lr=1e-3
python3 scripts/train.py --exp 07  # VGG-A-BN lr=2e-3
```

#### 2. 结果可视化
```bash
# 生成实验结果汇总表
python3 scripts/summarize_results.py

# 生成损失景观对比图
python3 scripts/plot_combined_loss_landscape.py

# 生成分层梯度对比图
python3 scripts/plot_both_gradients.py
```

## 实验结果总结
### 任务一：CIFAR-10分类
- 最优模型：卷积滤波器128的Original_BN_Dropout（12号实验和50号实验）
- 最佳验证准确率：89.86%
- 次优模型：默认配置的Original_BN_Dropout，准确率89.67%，与最优非常接近且参数量仅为最优的1/4，完全可以胜任分类任务

### 任务二：Batch Normalization研究
- BN平均提升准确率：6.99%
- 最优模型：VGG-A-BN（lr=1e-3）
- 最佳验证准确率：90.08%

## 重要说明
### 数据集
本项目使用CIFAR-10标准分类数据集：
- 所有训练脚本已集成自动下载功能，首次运行时会自动下载至对应任务的`./data`目录
- 自行下载地址（官方）：https://www.cs.toronto.edu/~kriz/cifar.html
- 我的数据集地址：https://drive.google.com/drive/folders/192fsnsRYRcfefc0776WxIC-81DsuKssV?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto
- 如选择手动下载，请在下载后请将数据集保存至对应任务的`./data`目录，并将loaders.py中查找download=True改为download=False

### 模型权重
训练好的模型权重会自动保存到对应实验的`reports/[实验名]/models/best_model.pth`路径下。

我训练的模型已经放在https://drive.google.com/drive/folders/192fsnsRYRcfefc0776WxIC-81DsuKssV?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto，下载后放入对应实验的`reports/[实验名]/models/`路径下即可。
