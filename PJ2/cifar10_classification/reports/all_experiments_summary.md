# 实验结果汇总（按测试准确率排序）

| ID | 模型 | 学习率 | 激活 | 优化器 | 最佳验证准确率 | 最终测试准确率 | 参数量 | 说明 |
|---|---|---|---|---|---|---|---|---|
| 54 | original_residual_bn | 1.0 | gelu | adadelta | 94.72% | 94.45% | 45,445,258 | BN+残差 | 卷积128 | 标签平滑0.1 | Adadelta |
| 56 | original_residual_bn | 1.0 | gelu | adadelta | 94.18% | 94.01% | 12,753,994 | BN+残差 | 全连接1024 | 标签平滑0.1 | Adadelta |
| 51 | original_residual_bn | 1.0 | gelu | adadelta | 94.06% | 93.51% | 45,445,258 | BN+残差 | 卷积128 | 标签平滑0.1 | Adadelta |
| 52 | original_residual_bn | 1.0 | gelu | adadelta | 93.22% | 93.23% | 45,445,258 | BN+残差 | 卷积128 | Adadelta |
| 55 | original_residual_bn | 1.0 | gelu | adadelta | 93.02% | 92.89% | 12,753,994 | BN+残差 | 全连接1024 | 标签平滑0.1 | Adadelta |
| 53 | original_residual_bn | 1.0 | gelu | adadelta | 92.52% | 92.47% | 45,445,258 | BN+残差 | 卷积128 | 权重衰减1e-5 | Adadelta |
| 44 | original_residual_bn | 1.0 | relu | adadelta | 92.38% | 92.16% | 11,699,274 | 优化器Adadelta |
| 41 | original_residual_bn | 0.001 | relu | adamw | 91.76% | 91.91% | 11,699,274 | 优化器AdamW |
| 12 | original_residual_bn | 0.001 | relu | adam | 91.52% | 91.79% | 45,445,258 | 卷积滤波器128 |
| 13 | original_residual_bn | 0.001 | relu | adam | 91.64% | 91.79% | 12,753,994 | 全连接层1024 |
| 34 | original_residual_bn | 0.001 | gelu | adam | 91.80% | 91.69% | 11,699,274 | 激活函数GELU |
| 23 | original_residual_bn | 0.001 | relu | adam | 92.00% | 91.59% | 11,699,274 | 标签平滑 |
| 03 | original_residual_bn | 0.001 | relu | adam | 91.28% | 91.55% | 11,699,274 | Original+BN+残差 |
| 40 | original_residual_bn | 0.001 | relu | adam | 91.28% | 91.55% | 11,699,274 | 优化器Adam |
| 35 | original_residual_bn | 0.001 | leaky_relu | adam | 90.88% | 91.14% | 11,699,274 | 激活函数LeakyReLU |
| 33 | original_residual_bn | 0.001 | swish | adam | 90.46% | 90.90% | 11,699,274 | 激活函数Swish |
| 11 | original_residual_bn | 0.001 | relu | adam | 89.90% | 90.30% | 3,194,410 | 卷积滤波器32 |
| 21 | original_residual_bn | 0.001 | relu | adam | 90.44% | 90.30% | 11,699,274 | Focal Loss |
| 42 | original_residual_bn | 0.001 | relu | rmsprop | 90.50% | 90.14% | 11,699,274 | 优化器RMSprop |
| 22 | original_residual_bn | 0.001 | relu | adam | 89.82% | 90.05% | 11,699,274 | 高L2正则 |
| 01 | original_bn | 0.001 | relu | adam | 89.08% | 88.94% | 9,753,674 | Original+BN |
| 02 | original_bn_dropout | 0.001 | relu | adam | 89.04% | 88.79% | 9,753,674 | Original+BN+Dropout（p=0.1） |
| 43 | original_residual_bn | 0.01 | relu | adagrad | 88.20% | 88.17% | 11,699,274 | 优化器Adagrad |
| 45 | original_residual_bn | 0.05 | relu | custom_sgd | 88.04% | 87.86% | 11,699,274 | 自定义SGD |
| 46 | original_residual_bn | 0.001 | relu | custom_adam | 87.46% | 87.40% | 11,699,274 | 自定义Adam |
| 32 | original_residual_bn | 0.001 | tanh | adam | 83.38% | 83.69% | 11,699,274 | 激活函数Tanh |
| 31 | original_residual_bn | 0.001 | sigmoid | adam | 34.76% | 34.90% | 11,699,274 | 激活函数Sigmoid |
