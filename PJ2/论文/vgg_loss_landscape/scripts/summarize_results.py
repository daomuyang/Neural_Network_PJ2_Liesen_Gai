import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_config_file(config_path):
    """从config.txt中读取最终测试结果"""
    test_acc = 0.0
    test_error = 0.0
    config_best_val_acc = 0.0
    
    if not os.path.exists(config_path):
        return test_acc, test_error, config_best_val_acc
    
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Best Validation Accuracy:'):
                config_best_val_acc = float(line.split(':')[1].strip().replace('%', ''))
            elif line.startswith('Final Test Accuracy:'):
                test_acc = float(line.split(':')[1].strip().replace('%', ''))
            elif line.startswith('Final Test Error:'):
                test_error = float(line.split(':')[1].strip().replace('%', ''))
    
    return test_acc, test_error, config_best_val_acc

def get_experiment_results(experiment_name):
    """加载完整实验结果"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_root = os.path.join(project_root, 'reports', experiment_name)
    curves_path = os.path.join(exp_root, 'metrics', 'training_curves.npz')
    config_path = os.path.join(exp_root, 'config.txt')
    
    # 读取训练曲线数据
    data = np.load(curves_path)
    train_loss = data.get('train_loss', np.array([]))
    val_loss = data.get('val_loss', np.array([]))
    train_acc = data.get('train_acc', np.array([]))
    val_acc = data.get('val_acc', np.array([]))
    
    # 读取最终测试结果
    final_test_acc, final_test_error, config_best_val_acc = parse_config_file(config_path)
    
    # 核心指标计算
    if len(val_acc) > 0:
        best_epoch_idx = np.argmax(val_acc)  # 最优epoch索引（0开始）
        best_val_acc = val_acc[best_epoch_idx]
        best_epoch = best_epoch_idx + 1     # 转换为1开始的epoch数
        best_train_acc = train_acc[best_epoch_idx]
        best_train_loss = train_loss[best_epoch_idx]
        best_val_loss = val_loss[best_epoch_idx]
        
        overfit = best_train_acc - best_val_acc
    else:
        best_val_acc = config_best_val_acc
        best_epoch = 0
        best_train_acc = 0.0
        best_train_loss = 0.0
        best_val_loss = 0.0
        overfit = 0.0
    
    # 保留最后一轮指标用于对比
    final_train_loss = train_loss[-1] if len(train_loss) > 0 else 0.0
    final_val_loss = val_loss[-1] if len(val_loss) > 0 else 0.0
    final_train_acc = train_acc[-1] if len(train_acc) > 0 else 0.0
    final_val_acc = val_acc[-1] if len(val_acc) > 0 else 0.0
    val_error = 100.0 - final_val_acc
    
    return {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'best_train_acc': best_train_acc,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'final_test_acc': final_test_acc,
        'val_error': val_error,
        'test_error': final_test_error,
        'overfit': overfit  
    }

if __name__ == '__main__':
    experiments = [
        # ========== 固定学习率实验（0-7号，用于Loss Landscape） ==========
        ('VGG-A', '1e-4', 'vgg_a_lr0001'),
        ('VGG-A', '5e-4', 'vgg_a_lr0005'),
        ('VGG-A', '1e-3', 'vgg_a_lr001'),
        ('VGG-A', '2e-3', 'vgg_a_lr002'),
        ('VGG-A-BN', '1e-4', 'vgg_a_bn_lr0001'),
        ('VGG-A-BN', '5e-4', 'vgg_a_bn_lr0005'),
        ('VGG-A-BN', '1e-3', 'vgg_a_bn_lr001'),
        ('VGG-A-BN', '2e-3', 'vgg_a_bn_lr002'),
        
        # ========== 最优性能实验（8-9号，带调度器，用于最终报告） ==========
        ('VGG-A (Best)', '-', 'vgg_a_best_performance'),
        ('VGG-A-BN (Best)', '-', 'vgg_a_bn_best_performance'),
    ]

    all_results = []
    for model, lr, exp_name in experiments:
        try:
            res = get_experiment_results(exp_name)
            all_results.append((model, lr, res))
            print(f"✅ 加载成功: {exp_name} | 过拟合: {res['overfit']:.2f}%")
        except Exception as e:
            print(f"⚠️ 加载失败: {exp_name} | 错误: {str(e)}")
            continue

    # 分离不同类型的实验
    fixed_lr_results = all_results[:8]
    best_results = all_results[8:]
    
    # 统计BN效果
    vgg_a_fixed = [res for m, lr, res in fixed_lr_results if m == 'VGG-A']
    vgg_a_bn_fixed = [res for m, lr, res in fixed_lr_results if m == 'VGG-A-BN']
    
    vgg_a_avg_acc = np.mean([r['best_val_acc'] for r in vgg_a_fixed]) if vgg_a_fixed else 0.0
    vgg_a_bn_avg_acc = np.mean([r['best_val_acc'] for r in vgg_a_bn_fixed]) if vgg_a_bn_fixed else 0.0
    avg_improve = vgg_a_bn_avg_acc - vgg_a_avg_acc
    
    vgg_a_avg_epoch = np.mean([r['best_epoch'] for r in vgg_a_fixed]) if vgg_a_fixed else 0.0
    vgg_a_bn_avg_epoch = np.mean([r['best_epoch'] for r in vgg_a_bn_fixed]) if vgg_a_bn_fixed else 0.0
    avg_epoch_improve = vgg_a_avg_epoch - vgg_a_bn_avg_epoch
    
    vgg_a_avg_overfit = np.mean([r['overfit'] for r in vgg_a_fixed]) if vgg_a_fixed else 0.0
    vgg_a_bn_avg_overfit = np.mean([r['overfit'] for r in vgg_a_bn_fixed]) if vgg_a_bn_fixed else 0.0
    avg_overfit_reduce = vgg_a_avg_overfit - vgg_a_bn_avg_overfit

    # 保存结果
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(project_root, 'reports', 'experiment_results.txt')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*130 + "\n")
        f.write("                VGG Batch Normalization 实验结果汇总\n")
        f.write("="*130 + "\n\n")
        
        # 第一部分：固定学习率实验结果
        f.write("【第一部分：固定学习率实验（用于Loss Landscape可视化）】\n")
        f.write("-"*130 + "\n")
        header = (
            "Model        LR    Best Val Acc(%)  Best Epoch  Best Train Acc(%)  Final Test Acc(%)  Test Error(%)  Overfit(%)\n"
        )
        f.write(header)
        f.write("-"*130 + "\n")
        
        for model, lr, res in fixed_lr_results:
            row = (
                f"{model:<12} {lr:<6} {res['best_val_acc']:>14.2f} {res['best_epoch']:>12d} "
                f"{res['best_train_acc']:>17.2f} {res['final_test_acc']:>17.2f} "
                f"{res['test_error']:>14.2f} {res['overfit']:>12.2f}\n"
            )
            f.write(row)
        
        f.write("-"*130 + "\n\n")
        
        # 第二部分：最优性能实验结果
        f.write("【第二部分：最优性能实验（带CosineAnnealingLR调度器）】\n")
        f.write("-"*130 + "\n")
        f.write(header)
        f.write("-"*130 + "\n")
        
        for model, lr, res in best_results:
            row = (
                f"{model:<12} {lr:<6} {res['best_val_acc']:>14.2f} {res['best_epoch']:>12d} "
                f"{res['best_train_acc']:>17.2f} {res['final_test_acc']:>17.2f} "
                f"{res['test_error']:>14.2f} {res['overfit']:>12.2f}\n"
            )
            f.write(row)
        
        f.write("-"*130 + "\n\n")
        
        # 第三部分：BN效果总结
        f.write("【第三部分：Batch Normalization 效果总结】\n")
        f.write("-"*130 + "\n")
        f.write(f"平均最佳验证精度提升: {avg_improve:.2f}%\n")
        f.write(f"平均达到最优轮数速度提升: {avg_epoch_improve:.1f} 轮\n")
        f.write(f"平均过拟合程度减少: {avg_overfit_reduce:.2f}%\n\n")
        
        f.write(f"无BN模型最优测试精度: {best_results[0][2]['final_test_acc']:.2f}% (测试误差: {best_results[0][2]['test_error']:.2f}%)\n")
        f.write(f"无BN模型最优过拟合: {best_results[0][2]['overfit']:.2f}%\n\n")
        
        f.write(f"有BN模型最优测试精度: {best_results[1][2]['final_test_acc']:.2f}% (测试误差: {best_results[1][2]['test_error']:.2f}%)\n")
        f.write(f"有BN模型最优过拟合: {best_results[1][2]['overfit']:.2f}%\n\n")
        
        f.write(f"BN最终精度提升: {best_results[1][2]['final_test_acc'] - best_results[0][2]['final_test_acc']:.2f}%\n")
        f.write(f"BN过拟合减少: {best_results[0][2]['overfit'] - best_results[1][2]['overfit']:.2f}%\n")
        
    
    print("\n" + "="*130)
    print("✅ 所有实验结果汇总完成！")
    print(f"📄 结果文件: {save_path}")
    print(f"📊 无BN最优测试精度: {best_results[0][2]['final_test_acc']:.2f}%")
    print(f"📊 有BN最优测试精度: {best_results[1][2]['final_test_acc']:.2f}%")
    print(f"✨ BN最终精度提升: {best_results[1][2]['final_test_acc'] - best_results[0][2]['final_test_acc']:.2f}%")
    print(f"📉 BN平均过拟合减少: {avg_overfit_reduce:.2f}%")
    print("="*130)