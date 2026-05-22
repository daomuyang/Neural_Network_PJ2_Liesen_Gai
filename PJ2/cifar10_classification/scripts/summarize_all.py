import os

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_root = os.path.join(project_root, 'reports')
    output_path = os.path.join(reports_root, 'all_experiments_summary.md')

    experiments = []

    for exp_name in sorted(os.listdir(reports_root)):
        exp_path = os.path.join(reports_root, exp_name)
        if not os.path.isdir(exp_path):
            continue

        config_path = os.path.join(exp_path, 'metrics', 'config.txt')
        if not os.path.exists(config_path):
            continue

        config = {}
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ': ' in line:
                    k, v = line.split(': ', 1)
                    config[k] = v

        exp_id = exp_name.split('_')[0] if '_' in exp_name else exp_name

        # 验证准确率
        try:
            best_val_acc = float(config.get('Best Validation Accuracy', '0').replace('%', ''))
        except:
            best_val_acc = 0.0

        # 测试准确率
        try:
            final_test_acc = float(config.get('Final Test Accuracy', '0').replace('%', ''))
        except:
            final_test_acc = 0.0

        exp_info = {
            'id': exp_id,
            'name': exp_name,
            'model': config.get('model', 'unknown'),
            'lr': config.get('lr', 'unknown'),
            'activation': config.get('activation', 'relu'),
            'optimizer': config.get('optimizer', 'adam'),
            'best_val_acc': config.get('Best Validation Accuracy', 'unknown'),
            'final_test_acc': config.get('Final Test Accuracy', 'unknown'),
            'test_acc_val': final_test_acc,
            'params': config.get('Number of Parameters', 'unknown'),
            'note': config.get('note', '')
        }
        experiments.append(exp_info)

    # 按测试准确率从高到低排序
    experiments.sort(key=lambda x: x['test_acc_val'], reverse=True)

    md = "# 实验结果汇总（按测试准确率排序）\n\n"
    md += "| ID | 模型 | 学习率 | 激活 | 优化器 | 最佳验证准确率 | 最终测试准确率 | 参数量 | 说明 |\n"
    md += "|---|---|---|---|---|---|---|---|---|\n"

    for e in experiments:
        md += f"| {e['id']} | {e['model']} | {e['lr']} | {e['activation']} | {e['optimizer']} | {e['best_val_acc']} | {e['final_test_acc']} | {e['params']} | {e['note']} |\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"✅ 汇总完成：{output_path}")

if __name__ == '__main__':
    main()