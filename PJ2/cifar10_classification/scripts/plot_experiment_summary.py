import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

YAML_PATH = Path("./configs/experiments.yaml")
RESULTS_ROOT = Path("./reports")
FIG_SAVE_ROOT = Path("./reports/figures")

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    
    # 线条和网格样式
    'lines.linewidth': 2.5,
    'lines.markersize': 4,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.4,
    'grid.color': '#cccccc',
    
    # 边框和布局
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'figure.figsize': (14, 10),
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

PROFESSIONAL_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# 子图配置
SUBPLOT_CONFIG = {
    "train_loss": {"title": "Training Loss", "ylabel": "Loss"},
    "val_loss": {"title": "Validation Loss", "ylabel": "Loss"},
    "train_acc": {"title": "Training Accuracy", "ylabel": "Accuracy (%)"},
    "val_acc": {"title": "Validation Accuracy", "ylabel": "Accuracy (%)"}
}

def load_experiment_config():
    """加载实验配置，按系列分组"""
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    experiments = config['experiments']
    exp_series = {}
    
    for exp_id, exp_info in experiments.items():
        try:
            exp_id_int = int(exp_id)
        except ValueError:
            print(f"⚠️  跳过无效实验ID: {exp_id}")
            continue
        
        series_num = (exp_id_int // 10) * 10
        series_id = f"{series_num:02d}"
        
        if series_id not in exp_series:
            exp_series[series_id] = []
        
        exp_series[series_id].append({
            "exp_id": f"{exp_id_int:02d}",
            "name": exp_info["name"]
        })
    
    return exp_series

def load_training_curves(exp_name):
    """加载单个实验的训练曲线数据"""
    curves_path = RESULTS_ROOT / exp_name / "metrics" / "training_curves.npz"
    
    if not curves_path.exists():
        print(f"⚠️  跳过实验 {exp_name}: 未找到训练曲线文件")
        return None
    
    try:
        data = np.load(curves_path)
        required_keys = ["train_loss", "val_loss", "train_acc", "val_acc"]
        curves = {}
        
        for key in required_keys:
            if key not in data:
                print(f"⚠️  跳过实验 {exp_name}: 缺少 {key} 数据")
                return None
            curves[key] = data[key]
        
        curves["epoch"] = np.arange(1, len(curves["train_loss"]) + 1)
        return curves
        
    except Exception as e:
        print(f"⚠️  加载实验 {exp_name} 失败: {str(e)}")
        return None

def plot_single_series(series_id, experiments):
    """绘制单个系列汇总图"""
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    subplot_keys = list(SUBPLOT_CONFIG.keys())
    
    # 遍历该系列的所有实验
    for exp_idx, exp in enumerate(experiments):
        exp_name = exp["name"]
        curves = load_training_curves(exp_name)
        
        if curves is None:
            continue
        
        # 为每个子图绘制曲线
        for ax_idx, key in enumerate(subplot_keys):
            ax = axes[ax_idx]
            ax.plot(
                curves["epoch"],
                curves[key],
                color=PROFESSIONAL_COLORS[exp_idx % len(PROFESSIONAL_COLORS)],
                marker='o',
                markevery=5,  # 每5个epoch显示一个标记点
                alpha=0.9,
                label=exp_name 
            )
    
    # 配置每个子图
    for ax_idx, key in enumerate(subplot_keys):
        ax = axes[ax_idx]
        config = SUBPLOT_CONFIG[key]
        
        ax.set_title(config["title"], fontweight='bold', pad=12)
        ax.set_xlabel("Epoch", labelpad=8)
        ax.set_ylabel(config["ylabel"], labelpad=8)

        ax.legend(
            loc="lower right",
            frameon=True,
            edgecolor='#dddddd',
            fancybox=False,
            borderpad=0.8
        )

        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle(
        f"Experiment Series {series_id} Training Curves",
        fontsize=16,
        fontweight='bold',
        y=0.98,  
        ha='center'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    FIG_SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    save_path = FIG_SAVE_ROOT / f"{series_id}_series_summary.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ 系列 {series_id} 汇总图已保存: {save_path}")

def main():
    print("="*60)
    print("📊 实验训练曲线汇总图生成器")
    print("="*60)
    
    try:
        exp_series = load_experiment_config()
        print(f"\n成功加载 {len(exp_series)} 个实验系列:")
        for series_id, exps in exp_series.items():
            print(f"  - {series_id}系列: {len(exps)} 个实验")
    except Exception as e:
        print(f"❌ 加载实验配置失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n开始生成汇总图...\n")
    for series_id in sorted(exp_series.keys()):
        sorted_exps = sorted(exp_series[series_id], key=lambda x: int(x["exp_id"]))
        plot_single_series(series_id, sorted_exps)
    
    print("\n" + "="*60)
    print("🎉 所有汇总图生成完成！")
    print(f"📁 图片保存位置: {FIG_SAVE_ROOT.resolve()}")
    print("="*60)

if __name__ == "__main__":
    if not YAML_PATH.exists():
        print(f"❌ 错误: 实验配置文件不存在 {YAML_PATH}")
        exit(1)
    
    if not RESULTS_ROOT.exists():
        print(f"❌ 错误: 实验结果目录不存在 {RESULTS_ROOT}")
        exit(1)
    
    main()