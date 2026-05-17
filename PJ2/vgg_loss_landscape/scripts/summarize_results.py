import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_experiment_results(experiment_name):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    curves_path = os.path.join(project_root, 'reports', experiment_name, 'metrics', 'training_curves.npz')
    
    data = np.load(curves_path)
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    train_acc = data['train_acc']
    val_acc = data['val_acc']
    
    best_val_acc = np.max(val_acc)
    best_epoch = np.argmax(val_acc) + 1
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    overfit = final_train_acc - final_val_acc
    
    return {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'overfit': overfit
    }

if __name__ == '__main__':
    experiments = [
        ('VGG-A', '2e-3', 'vgg_a_lr002'),
        ('VGG-A', '1e-3', 'vgg_a_lr001'),
        ('VGG-A', '5e-4', 'vgg_a_lr0005'),
        ('VGG-A', '1e-4', 'vgg_a_lr0001'),
        ('VGG-A-BN', '2e-3', 'vgg_a_bn_lr002'),
        ('VGG-A-BN', '1e-3', 'vgg_a_bn_lr001'),
        ('VGG-A-BN', '5e-4', 'vgg_a_bn_lr0005'),
        ('VGG-A-BN', '1e-4', 'vgg_a_bn_lr0001'),
    ]

    all_results = []
    for model, lr, exp_name in experiments:
        try:
            res = get_experiment_results(exp_name)
            all_results.append((model, lr, res))
        except Exception:
            continue

    vgg_a_accs = [res['best_val_acc'] for m, lr, res in all_results if m == 'VGG-A']
    vgg_a_bn_accs = [res['best_val_acc'] for m, lr, res in all_results if m == 'VGG-A-BN']
    vgg_a_avg = np.mean(vgg_a_accs) if vgg_a_accs else 0
    vgg_a_bn_avg = np.mean(vgg_a_bn_accs) if vgg_a_bn_accs else 0
    avg_improve = vgg_a_bn_avg - vgg_a_avg
    best_exp = max(all_results, key=lambda x: x[2]['best_val_acc']) if all_results else None

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(project_root, 'reports', 'experiment_results.txt')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=== VGG BatchNorm Experiment Results ===\n")
        f.write("Model    LR    Best Acc(%)  Best Epoch  Final Train Loss  Final Val Loss  Final Train Acc(%)  Final Val Acc(%)  Overfit(%)\n")
        f.write("-"*140 + "\n")
        
        for model, lr, res in all_results:
            f.write(f"{model:<8} {lr:<6} {res['best_val_acc']:>8.2f} {res['best_epoch']:>12d} {res['final_train_loss']:>16.4f} {res['final_val_loss']:>14.4f} {res['final_train_acc']:>18.2f} {res['final_val_acc']:>16.2f} {res['overfit']:>12.2f}\n")
        
        f.write("-"*140 + "\n")
        f.write(f"\nAverage Best Accuracy:\n")
        f.write(f"VGG-A:        {vgg_a_avg:.2f}%\n")
        f.write(f"VGG-A-BN:     {vgg_a_bn_avg:.2f}%\n")
        f.write(f"BN Improvement:{avg_improve:.2f}%\n\n")
        f.write(f"Best Experiment: {best_exp[0]} (lr={best_exp[1]}), Best Val Acc: {best_exp[2]['best_val_acc']:.2f}%\n")