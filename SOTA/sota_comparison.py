# sota_comparison.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_performance_bar_chart(metrics_dict):
    """
    绘制各方法在 Accuracy、AUC、F1 指标上的对比柱状图。
    """
    # 将字典转为 DataFrame，行索引为方法名称，列为各指标
    df = pd.DataFrame(metrics_dict).T
    metrics = df.columns.tolist()
    methods = df.index.tolist()

    # 创建一个 figure，每个指标单独一个子图
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(methods, df[metric], color=['blue', 'orange', 'green', 'red'][:len(methods)])
        ax.set_title(f"Comparison of {metric}")
        ax.set_xlabel("Method")
        ax.set_ylabel(metric)
        # 在柱子上标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}",
                    ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("sota_performance_comparison.png")
    plt.show()

def main():
    # 定义各方法的性能指标，注意将下列示例数值替换为你的真实实验结果
    metrics_dict = {
        "PGCN": {"Accuracy": 0.7273, "AUC": 0.6667, "F1": 0.6667},
        "MethodA": {"Accuracy": 0.8182, "AUC": 0.7500, "F1": 0.0333},
        "MethodB": {"Accuracy": 0.6364, "AUC": 0.6000, "F1": 0.6333},
        "MethodC": {"Accuracy": 0.4545, "AUC": 0.0000, "F1": 0.4167}
    }

    # 绘制对比柱状图
    plot_performance_bar_chart(metrics_dict)



if __name__ == "__main__":
    main()