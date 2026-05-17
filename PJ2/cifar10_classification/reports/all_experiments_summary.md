# 实验结果汇总

| ID | 模型 | 学习率 | 激活 | 优化器 | 最佳准确率 | 参数量 | 说明 |
|---|---|---|---|---|---|---|---|
| 12 | original_bn_dropout | 0.001 | relu | adam | 89.86% | 37,676,554 | 基线+卷积滤波器数量翻倍（128） |
| 50 | original_bn_dropout | 0.001 | relu | adam | 89.86% | 37,676,554 | 组合模型1：滤波器128 + L2 + Dropout(0.1) + Adam |
| 51 | original_bn_dropout | 0.001 | relu | adam | 89.76% | 37,676,554 | 组合模型2：滤波器128+L2+Dropout(0.1)+Adam+标签平滑0.1 |
| 02 | original_bn_dropout | 0.001 | relu | adam | 89.67% | 9,756,426 | Original+BN+Dropout（p=0.1） |
| 01 | original_bn | 0.001 | relu | adam | 89.60% | 9,756,426 | Original+BN |
| 23 | original_bn_dropout | 0.001 | relu | adam | 89.42% | 9,756,426 | 基线+标签平滑（系数0.1） |
| 03 | original_residual_bn | 0.001 | relu | adam | 89.33% | 20,547,082 | Original+BN+残差 |
| 33 | original_bn_dropout | 0.001 | swish | adam | 89.20% | 9,756,426 | 激活函数消融：Swish |
| 22 | original_bn_dropout | 0.001 | relu | adam | 89.15% | 9,756,426 | 基线+高强度L2正则化（5e-4） |
| 13 | original_bn_dropout | 0.001 | relu | adam | 89.00% | 10,811,146 | 基线+全连接层神经元翻倍（1024） |
| 21 | original_bn_dropout | 0.001 | relu | adam | 88.33% | 9,756,426 | 基线+Focal Loss+L2+标签平滑（0.1） |
| 11 | original_bn_dropout | 0.001 | relu | adam | 87.06% | 2,708,362 | 基线+卷积滤波器数量减半（32） |
| 32 | original_bn_dropout | 0.001 | tanh | adam | 85.81% | 9,756,426 | 激活函数消融：Tanh |
| 00 | original | 0.001 | relu | adam | 84.81% | 9,750,922 | 纯Original基线（统一Adam+0.001） |
| 42 | original_bn_dropout | 0.001 | relu | custom_adam | 84.71% | 9,756,426 | 基线+手写Adam优化器 |
| 41 | original_bn_dropout | 0.05 | relu | custom_sgd | 64.81% | 9,756,426 | 基线+手写SGD优化器 |
| 31 | original_bn_dropout | 0.001 | sigmoid | adam | 34.81% | 9,756,426 | 激活函数消融：Sigmoid |
