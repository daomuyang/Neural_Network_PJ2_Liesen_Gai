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
├── cifar10_classification/ # 任务一：CIFAR-10图像分类实验
│   ├── utils/ # 通用工具函数
│   │   ├── __init__.py
│   │   ├── train_utils.py # 训练工具：准确率、种子、目录创建
│   │   └── custom_optimizers.py # 自定义优化器
│   ├── models/ # 模型定义
│   │   ├── __init__.py
│   │   └── models.py # 任务一模型：Original、BN、Dropout、残差模块
│   ├── configs/ # 实验配置
│   │   └── experiments.yaml # 超参数配置
│   ├── scripts/ # 可执行脚本
│   │   ├── train.py # 训练主脚本
│   │   ├── summarize_all.py # 结果汇总
│   │   ├── evl_only.py # 测试专用脚本
│   │   ├── plot_experiment_summary.py # 各系列合并训练图像绘制
│   │   ├── grad_cam.py # Grad-CAM可视化
│   │   └── visualize_filters.py # 滤波器可视化
│   ├── data/ # 数据集目录
│   │   ├── __init__.py
│   │   └── loaders.py # 数据加载与预处理
│   └── reports/ # 实验输出目录
│       ├── all_experiments_summary.md # 实验汇总
│       ├── figures/ # 保存各系列合并训练图像
│       └── [实验文件夹]/ # 单个实验输出
│           ├── visualizations/ # grad-cam与卷积核可视化
│           │   ├── filters/ # 第一层部分卷积核可视化
│           │   └── gradcam/ # grad-cam热力图可视化
│           ├── metrics/ # 训练指标
│           │   ├── config.txt # 配置+最优精度+参数量
│           │   ├── losses.txt # 训练损失
│           │   └── training_curves.npz # 损失与精度曲线
│           └── figures/ # 实验图表
│               └── training_curve.png # 训练曲线
├── vgg_loss_landscape/ # 任务二：BN机制研究实验
│   ├── utils/ # 通用工具函数
│   │   ├── __init__.py
│   │   └── train_utils.py # 训练工具
│   ├── models/ # 模型定义
│   │   ├── __init__.py
│   │   ├── vgg.py # VGG-A、VGG-A-BN
│   │   └── nn.py # 基础网络组件
│   ├── configs/ # 实验配置
│   │   └── experiments.yaml # 超参数配置
│   ├── scripts/ # 可执行脚本
│   │   ├── train.py # 训练主脚本
│   │   ├── summarize_results.py # 结果汇总
│   │   ├── plot_both_gradients.py # 梯度范数绘图
│   │   └── plot_combined_loss_landscape.py # 损失景观绘图
│   ├── data/ # 数据集目录
│   │   ├── __init__.py
│   │   └── loaders.py # 数据加载
│   └── reports/ # 实验输出目录
│       ├── experiment_results.txt # 实验结果汇总
│       ├── gradient_result.txt # 梯度结果汇总
│       ├── figures/ # 全局图表
│       └── [实验文件夹]/ # 单个实验输出
│           ├── config.txt # 实验配置
│           ├── metrics/ # 训练指标
│           │   ├── losses.txt # 训练损失
│           │   ├── training_curves.npz # 曲线数据
│           │   ├── grad_last_layer_avg.npy # 最后一层梯度
│           │   └── grad_mid_layer_avg.npy # 中间层梯度
│           └── figures/ # 实验图表
│               └── training_curve.png # 训练曲线
└── papers/ # 参考文献
    ├── bib.bib # 参考文献库
    ├── Gaussian Error Linear Units.pdf # GELU论文
    ├── How does Batch Normalization help optimization.pdf # BN优化机制
    ├── Deep_Residual_Learning_for_Image_Recognition.pdf # ResNet论文
    ├── Grad-CAM_Visual_Explanations_from_Deep_Networks_via_Gradient-Based_Localization.pdf # Grad-CAM
    └── Decoupled Weight Decay Regularization.pdf # 解耦权重衰减
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
- PyTorch版本：2.9.1+cu128

### 依赖安装
```bash
pip install torch==2.9.1 torchvision==0.20.1
pip install numpy==1.26.4 matplotlib==3.9.0 pyyaml==6.0.1 tqdm==4.66.4
```

### 硬件要求
- 推荐配置：NVIDIA GPU（显存≥8GB）

## 运行指南
**所有命令均在对应任务目录下执行，请勿在PJ2根目录下运行**

### 任务一：CIFAR-10分类实验
**所有命令均在`cifar10_classification`目录下执行**

#### 1. 训练实验
```bash
# 运行单个实验（--exp指定实验编号）
python3 scripts/train.py --exp 01  # BN模型
python3 scripts/train.py --exp 02  # BN+Dropout模型
python3 scripts/train.py --exp 54  # 全局最优组合模型（残差+BN）
```

#### 2. 单独测试已训练模型
```bash
# 手动修改scripts/evl_only.py中的EXP_ID为目标实验序号
# 确保最优模型已保存在对应实验的models/best_model.pth路径下
python3 scripts/evl_only.py
```

#### 3. 结果可视化
```bash
# 生成所有实验结果汇总表，输出至reports/all_experiments_summary.md
python3 scripts/summarize_all.py

# 生成各系列合并训练曲线，输出至reports/figures/
python3 scripts/plot_experiment_summary.py

# 可视化所有实验的卷积滤波器，输出至对应实验的visualizations/filters/
python3 scripts/visualize_filters.py

# 生成所有实验的Grad-CAM热力图，输出至对应实验的visualizations/gradcam/
python3 scripts/grad_cam.py
```

### 任务二：Batch Normalization研究
**所有命令均在`vgg_loss_landscape`目录下执行**

#### 1. 训练实验
```bash
# 0-7号：固定学习率实验（用于损失景观可视化）
python3 scripts/train.py --exp 00  # VGG-A lr=1e-4
python3 scripts/train.py --exp 01  # VGG-A lr=5e-4
python3 scripts/train.py --exp 02  # VGG-A lr=1e-3
python3 scripts/train.py --exp 03  # VGG-A lr=2e-3
python3 scripts/train.py --exp 04  # VGG-A-BN lr=1e-4
python3 scripts/train.py --exp 05  # VGG-A-BN lr=5e-4
python3 scripts/train.py --exp 06  # VGG-A-BN lr=1e-3
python3 scripts/train.py --exp 07  # VGG-A-BN lr=2e-3

# 8-9号：带余弦退火调度器的最优性能对比实验
python3 scripts/train.py --exp 8   # VGG-A 相对最优配置（lr=1e-3 + CosineAnnealingLR）
python3 scripts/train.py --exp 9   # VGG-A-BN 最优配置（lr=1e-3 + CosineAnnealingLR）
```

#### 2. 结果可视化（所有脚本无需指定实验序号）
```bash
# 生成完整实验结果汇总表，输出至reports/experiment_results.txt
python3 scripts/summarize_results.py

# 生成损失景观对比图，输出至reports/figures/
python3 scripts/plot_combined_loss_landscape.py

# 自动读取lr=0.001实验的梯度数据
# 生成分层梯度对比图（输出至reports/figures/）和梯度结果汇总（输出至reports/gradient_result.txt）
python3 scripts/plot_both_gradients.py
```

## 实验结果总结
### 任务一：CIFAR-10分类
- 最优大模型：54号
- 最佳测试准确率：94.45%（50轮）、93.51%（30轮）
- 最优小模型：56号（参数量仅为54号的1/4）
- 最佳测试准确率：94.01%（50轮）、92.89%（30轮）

### 任务二：Batch Normalization研究
- 固定学习率下，BN平均提升验证准确率：4.43%
- 加入学习率调度器后，最优测试准确率达88.28%（较同条件无BN模型提升3.54%）
- 30轮内平均达到最优轮数速度提升：2.2轮，最优实验过拟合程度降低0.61%
- 核心机制：将中间卷积层稳态梯度范数降低76.7%、最后分类层最大梯度范数提升19.4%，通过平滑损失景观、调控分层梯度、增强学习率鲁棒性优化训练过程

## 重要说明
### 数据集
本项目使用CIFAR-10标准分类数据集：
- 所有训练脚本已集成自动下载功能，首次运行时会自动下载至对应任务的`./data`目录
- 自行下载地址（官方）：https://www.cs.toronto.edu/~kriz/cifar.html
- 我的数据集地址：https://drive.google.com/drive/folders/192fsnsRYRcfefc0776WxIC-81DsuKssV?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto
- 如选择手动下载，请将数据集保存至对应任务的`./data`目录，并将loaders.py中的`download=True`改为`download=False`

### 模型权重
训练好的模型权重会自动保存到对应实验的`reports/[实验名]/models/best_model.pth`路径下。

我训练的所有模型已上传至Google Drive：
https://drive.google.com/drive/folders/192fsnsRYRcfefc0776WxIC-81DsuKssV?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

包含：
- 任务一所有27组消融实验的最优模型
- 任务二所有8组固定学习率配置下的最优模型
- 任务二2组添加余弦退火调度器后最优配置下的模型
- 所有模型均为PyTorch标准`.pth`格式，可直接用于推理与验证
